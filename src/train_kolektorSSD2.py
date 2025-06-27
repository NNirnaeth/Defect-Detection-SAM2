import argparse
import glob
import json
import os
import random
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pycocotools import mask as coco_mask

from sklearn.model_selection import train_test_split

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# PREPARACION DEL DATASET

def prepare_dataset(dataset_name, data_dir="datasets", test_size=0.2):
    """
    Prepara un dataset en funcion de su tipo.

    Parametros:
    - dataset_name (str): Nombre del dataset ('kolektor', 'chest', etc.).
    - data_dir (str): Ruta base donde estan almacenados los datasets.
    - test_size (float): Proporcion del dataset que se usara para test.

    Retorna:
    - train_data (list): Lista de diccionarios con imagenes y anotaciones de entrenamiento.
    - test_data (list): Lista de diccionarios con imagenes y anotaciones de test.
    """
    train_data, test_data = [], []

    if dataset_name == "kolektor":
        subsets = ["train", "test", "val"]
        data_splits = {subset: [] for subset in subsets}

        for subset in subsets:
            subset_dir = os.path.join(data_dir, "kolektorSDD2", subset)

            # Buscar imagenes y mascaras en la misma carpeta
            image_files = glob.glob(os.path.join(subset_dir, "*.jpg"))
            json_files = glob.glob(os.path.join(subset_dir, "*.json"))

            print(f"Buscando imagenes en {subset_dir}: {len(image_files)} encontradas")
            print(f"Buscando anotaciones en {subset_dir}: {len(json_files)} encontradas")

            for image_path in image_files:

                # Verificar si el archivo IMAGE existe
                if not os.path.exists(image_path):
                    print(f" Advertencia: No se encontró Image para {image_path}. Ignorando.")
                    continue
                image_name = os.path.basename(image_path)

                # Verificar si el archivo JSON existe
                mask_path = os.path.join(subset_dir, image_name.replace(".jpg", ".json"))
                if not os.path.exists(mask_path):
                    print(f" Advertencia: No se encontró JSON para {image_path}. Ignorando.")
                    continue


                # Leer el JSON para verificar si tiene anotaciones
                with open(mask_path, "r") as f:
                    mask_data = json.load(f)

                # Verificar si el JSON contiene la clave "annotations"
                if "annotations" not in mask_data:
                    print(f" Error en {mask_path}: No tiene la clave 'annotations'. Ignorando.")
                    exit(0)
                    continue

                # Si tiene anotaciones, agregarlo al dataset
                if len(mask_data["annotations"]) > 0:
                    data_splits[subset].append({
                        "image": image_path,
                        "annotation": mask_path
                    })


        train_data = data_splits["train"]
        test_data = data_splits["test"]

        print(f"Total imágenes con anotaciones válidas en TRAIN: {len(train_data)}")
        print(f"Total imágenes con anotaciones válidas en TEST: {len(test_data)}")


    elif dataset_name == "chest":
        # Dataset de radiografias de pecho (Chest X-Ray)
        images_dir = os.path.join(data_dir, "chest", "images/images")
        masks_dir = os.path.join(data_dir, "chest", "masks/masks")
        csv_path = os.path.join(data_dir, "chest", "train.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontro el archivo CSV en {csv_path}")

        train_df = pd.read_csv(csv_path)
        train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=42)

        for _, row in train_df.iterrows():
            train_data.append({
                "image": os.path.join(images_dir, row['ImageId']),
                "annotation": os.path.join(masks_dir, row['MaskId'])
            })

        for _, row in test_df.iterrows():
            test_data.append({
                "image": os.path.join(images_dir, row['ImageId']),
                "annotation": os.path.join(masks_dir, row['MaskId'])
            })
    # Se puede aNadir aqui otro elif para un dataset nuevo

    else:
        raise ValueError(f"Dataset '{dataset_name}' no reconocido. Agrega su estructura en esta funcion.")

    print(f"Dataset '{dataset_name}' -> Train: {len(train_data)}, Test: {len(test_data)}")
    return train_data, test_data


# LECTURA Y PROCESAMIENTO DE IMAGENES - batch

def read_batch(data, base_path='', use_rle=False, visualize_data=False):
    ent = random.choice(data)

    img_path = os.path.join(base_path, ent["image"])
    Img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if Img is None:
        print(f"Error leyendo la imagen {img_path}")
        return None, None, None, 0

    # Cargar la mascara (RLE o imagen en escala de grises)
    if use_rle:
        ann_map = np.zeros((ent["image"]["height"], ent["image"]["width"]), dtype=np.uint8)
        for ann in ent["annotations"]:
            rle = ann["segmentation"]
            decoded_mask = coco_mask.decode(rle).astype(np.uint8)
            ann_map = np.maximum(ann_map, decoded_mask)
    else:
        ann_path = os.path.join(base_path, ent["annotation"])
        ann_map = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if ann_map is None:
            print(f"Error leyendo la mascara {ann_path}")
            return None, None, None, 0

    # Convertir a RGB
    Img = Img[..., ::-1]

    # Redimensionar
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
    img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # MAscara binaria
    binary_mask = (ann_map > 0).astype(np.uint8)
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)

    # Seleccionar puntos aleatorios dentro de la segmentacion
    coords = np.argwhere(eroded_mask > 0)
    points = np.array([coords[np.random.randint(len(coords))]] if len(coords) > 0 else [])

    binary_mask = np.expand_dims(binary_mask, axis=0)
    points = np.expand_dims(points, axis=1) if len(points) > 0 else np.zeros((0, 1, 2))

    # VISUALIZACION OPCIONAL

    if visualize_data:
        plt.figure(figsize=(15, 5))

        # Imagen original
        plt.subplot(1, 3, 1)
        plt.title('Imagen Original')
        plt.imshow(img)
        plt.axis('off')

        # Mascara binaria
        plt.subplot(1, 3, 2)
        plt.title('Mascara Binaria')
        plt.imshow(binary_mask[0], cmap='gray')
        plt.axis('off')

        # Mascara con puntos resaltados
        plt.subplot(1, 3, 3)
        plt.title('Mascara + Puntos')
        plt.imshow(binary_mask[0], cmap='gray')

        # Dibujar puntos seleccionados
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0][0], point[0][1], c=colors[i % len(colors)], s=100, label=f'Punto {i + 1}')

        plt.axis('off')
        plt.show()

    return img, binary_mask, points, len(ent["annotations"]) if use_rle else len(np.unique(ann_map)[1:])


# CARGAR Y REDIMENSIONAR IMAGEN Y MASCARA SOLO PARA LA EVALUACION (no extrae puntos ni convierte en binary mask)

def read_image(image_path, mask_path):
    """
    Carga y redimensiona una imagen y su máscara.
    - Si `mask_path` es una imagen (.png, .jpg), la carga directamente.
    - Si `mask_path` es un JSON (COCO format), decodifica la máscara.
    """
    # Cargar imagen con OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: No se pudo cargar la imagen en {image_path}. Verifica la ruta.")

    img = img[..., ::-1]  # Convertir BGR a RGB

    # Si la mascara es una imagen (.png, .jpg)
    if mask_path.endswith((".png", ".jpg", ".jpeg")):
        mask = cv2.imread(mask_path, 0)  # Cargar en escala de grises
        if mask is None:
            raise FileNotFoundError(f"Error: No se pudo cargar la máscara en {mask_path}. Verifica la ruta.")

    # Si la mascara es un json (COCO format)
    elif mask_path.endswith(".json"):
        with open(mask_path, "r") as f:
            mask_data = json.load(f)

        if "annotations" not in mask_data or len(mask_data["annotations"]) == 0:
            raise ValueError(f"Error: No se encontraron anotaciones en {mask_path}")

        # Crear una mascara vacia con las dimensiones de la imagen
        h, w = mask_data["image"]["height"], mask_data["image"]["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Decodificar cada anotacion y fusionarlas
        for ann in mask_data["annotations"]:
            if "segmentation" in ann:
                rle = ann["segmentation"]
                decoded_mask = coco_mask.decode(rle).astype(np.uint8)
                mask = np.maximum(mask, decoded_mask)  # Fusionar todas las máscaras

    else:
        raise ValueError(f"Formato de máscara desconocido: {mask_path}")

    # Redimensionar imagen y máscara
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    return img, mask


# DEFINIR EXACTAMENTE CUANTOS PUNTOS EXTRAER EN LA EVALUACION

def get_points(mask, num_points):  # Sample points inside the input mask
    points = []
    coords = np.argwhere(mask > 0)
    for i in range(num_points):
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)


# FUNCION DE ENTRENAMIENTO

def train(args, train_data):
    # Load model and predictor
    sam2_model = build_sam2(args.config, args.checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Configurate train
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # Configurate optimizer
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Configurate precision mixta
    scaler = torch.cuda.amp.GradScaler()

    # Use the arguments of argparse
    NO_OF_STEPS = args.steps
    FINE_TUNED_MODEL_NAME = args.model_name

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)  # 500 , 250, gamma = 0.1
    accumulation_steps = 4  # Number of steps to accumulate gradients before updating

    for step in range(1, NO_OF_STEPS + 1):
        with torch.cuda.amp.autocast():
            image, mask, input_point, num_masks = read_batch(train_data, visualize_data=False)
            if image is None or mask is None or num_masks == 0:
                continue

            input_label = np.ones((num_masks, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue

            if input_point.size == 0 or input_label.size == 0:
                continue

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None,
                                                                                    mask_logits=None,
                                                                                    normalize_coords=True)
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log(
                (1 - prd_mask) + 0.00001)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            # Apply gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()

            # Update scheduler
            scheduler.step()

            if step % 500 == 0:
                FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step) + ".torch"
                torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

            if step == 1:
                mean_iou = 0

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            if step % 100 == 0:
                print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)


# FUNCION DE EVALUACION CON UNA SOLA IMAGEN Y VISUALIZAR EL RESULTADO

def eval(args, test_data, num_samples=30):
    # Load base model SAM2
    sam2_model = build_sam2(args.config, args.checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Load the fine-tuned model
    fine_tuned_checkpoint = torch.load(args.fine_tuned_checkpoint, map_location="cuda")
    predictor.model.load_state_dict(fine_tuned_checkpoint)

    # Randomly select a test image from the test_data
    selected_entry = random.choice(test_data)
    image_path = selected_entry['image']
    mask_path = selected_entry['annotation']

    # Load the selected image and mask
    image, mask = read_image(image_path, mask_path)

    # Generate random points for the input
    input_points = get_points(mask, num_samples)

    # Perform inference and predict masks
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    # Process the predicted masks and sort by scores
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    # Initialize segmentation map and occupancy mask
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

    # Combine masks to create the final segmentation map
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue

        mask_bool = mask.astype(bool)
        mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
        seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
        occupancy_mask[mask_bool] = True  # Update occupancy_mask

    # Visualization: Show the original image, mask, and final segmentation side by side
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Test Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Original Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Final Segmentation')
    plt.imshow(seg_map, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# DEFINIR FUNCIONES PARA EVALUACION SAM2 CON test_data. CARGAR IMG/MASK,
# EXTRAER PUNTOS, CALCULAR COINCIDENCIA PREDICCION CON IOU.

# Convierte anotaciones en RLE a mascaras binarias.
def decode_rle_to_mask(rle):
    # rle = {"counts":"...", "size":[height,width]}
    return coco_mask.decode(rle).astype(np.uint8)


# Fusiona todas las anotaciones de una imagen en una sola mascara binaria.
def get_binary_gt_mask(entry, base_path=''):
    h, w = entry["image"]["height"], entry["image"]["width"]
    gt_mask = np.zeros((h, w), dtype=np.uint8)

    if "annotations" in entry:
        for ann in entry["annotations"]:
            decoded = decode_rle_to_mask(ann["segmentation"])
            gt_mask = np.maximum(gt_mask, decoded)

    return gt_mask


# Extrae puntos aleatorios dentro de la segmentacion.
def sample_points_from_mask(mask, num_points=30):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    chosen_idx = np.random.choice(len(coords), min(num_points, len(coords)), replace=False)
    pts = coords[chosen_idx]
    # Devuelve (N,2) en (y,x), predictor espera (N,2) en [y, x]
    return pts


# Calcula que tan bien la prediccion coincide con el ground truth (IoU)
def compute_iou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / float(union + 1e-6)


# EVALUAR TODAS LAS IMAGENES DE TEST_DATA Y CALCULAR METRICA DE RENDIMIENTO IOU

def evaluate_sam2(args, test_data, base_path='', iou_thresholds=[0.5, 0.75, 0.95]):
    # Construir modelo
    sam2_model = build_sam2(args.config, args.checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Cargar checkpoint fine-tuned
    ft_ckpt = torch.load(args.fine_tuned_checkpoint, map_location="cuda")
    predictor.model.load_state_dict(ft_ckpt)

    all_ious = []
    for entry in test_data:

        # Cargar imagen
        img_file = os.path.join(base_path, entry["image"]["file_name"])
        if not os.path.exists(img_file):
            print(f"Path does not exist: {img_file}")
            continue

        Img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if Img is None:
            print(f" No se pudo cargar la imagen: {img_file}")
            continue
        Img = Img[..., ::-1]  # BGR -> RGB


        # Obtener la mascara de Ground Truth
        try:
            # Ground-truth binaria combinada
            gt_mask = get_binary_gt_mask(entry, base_path)
        except Exception as ex:
            print(f"Cannot get binary GT mask: {entry}: {ex}")
            continue

        # Samplear puntos de la mascara GT
        points = sample_points_from_mask(gt_mask, num_points=30)
        if points is None:
            continue

        xx = np.ones((points.shape[0], 1), dtype=np.int32)
        # Inferencia SAM2
        with torch.no_grad():
            predictor.set_image(Img.copy())
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=np.ones((points.shape[0],), dtype=np.int32)
            )

        # Predecir varias mascaras por imagen
        # Ordenar mascaras por score y priorizar las mas precisas
        np_masks = np.array(masks[:, 0], dtype=np.uint8)
        np_scores = scores.flatten()  # Convert to a 1D array safely
        sorted_idxs = np.argsort(np_scores)[::-1]
        sorted_masks = np_masks[sorted_idxs]

        # Fusionar mascaras eliminando solapamientos
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)  #Inicializa mapa de segmentacion
        occupancy = np.zeros_like(seg_map, dtype=bool) # Mascara para evitar solapamientos
        for i, msk in enumerate(sorted_masks):
            m_bool = msk.astype(bool)  # Convertir mascara a booleano
            if (m_bool & occupancy).sum() / (msk.sum() + 1e-6) > 0.15:
                continue  # Si la superposicion es mayor al 15% ignorar mascara
            m_bool[occupancy] = False # Elimnar areas solapadas
            seg_map[m_bool] = i + 1 # Asignar mascara al mapa final
            occupancy[m_bool] = True # Actualizar ocupacion

        # Calcular IoU entre prediccion y ground truth
        # Para IoU, binarizamos la prediccion (1 si seg_map>0)
        pred_mask = (seg_map > 0).astype(np.uint8)
        iou_val = compute_iou(pred_mask, (gt_mask > 0).astype(np.uint8)) # Calcula IoU
        all_ious.append(iou_val)

    # Calcular metricas
    all_ious = np.array(all_ious)
    results = {}
    for th in iou_thresholds:
        results[f'IoU@{int(th * 100)}'] = np.mean(all_ious >= th)  # % de imagenes con IoU >= umbral

    # Imprimir estadisticos
    print("Evaluacion SAM2 en test set:")
    print(f"Promedio IoU: {all_ious.mean():.4f}")
    for k, v in results.items():
        print(f"{k}: {v * 100:.2f}%")

    return results


# ARGUMENTOS QUE VAN EN LA LINEA DE COMANDO AL EJECUTAR EL SCRIPT

def parse_args():
    # Argparse Configuration
    parser = argparse.ArgumentParser(description="Train or Evaluate SAM2 model.")
    parser.add_argument("mode", choices=["train", "eval", "evaluate_sam2"], help="Mode to run: 'train' or 'eval'")
    parser.add_argument("--dataset", type=str, choices=["kolektor", "chest"], required=True,
                        help="Dataset to use: 'kolektor' or 'chest'")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the SAM2 checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to the SAM2 configuration file.")

    # Train Parameters
    parser.add_argument("--steps", type=int, default=1500, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--model_name", type=str, default="fine_tuned_sam2", help="Name for the fine-tuned model.")

    # Evaluation Parameters
    parser.add_argument("--fine_tuned_checkpoint", type=str,
                        help="Path to the fine-tuned model checkpoint (required for eval).")

    return parser.parse_args()


# ASEGURAR QUE SE EJECUTA SOLO EN LA TERMINAL, NO CUANDO SE IMPORTA COMO MODULO EN OTRO SCRIPT

if __name__ == "__main__":

    # Read/parse arguments
    args = parse_args()

    # Seleccionar dataset según argumento `--dataset
    train_data, test_data = prepare_dataset(args.dataset, data_dir="datasets/")

    # Execute according to the chosen model
    if args.mode == "train":
        train(args, train_data)
    elif args.mode == "eval":
        if not args.fine_tuned_checkpoint:
            print("You must provide --fine_tuned_checkpoint for evaluation mode.")
        eval(args, test_data)
    elif args.mode == "evaluate_sam2":
        evaluate_sam2(args, test_data, base_path="datasets/kolektorSDD2/test/")
