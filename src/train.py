import os
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils import read_batch
from evaluate import evaluate_on_validation_set
from metrics_logger import MetricsLogger

def train(args, _):
    # Preparar split interno
    from dataloaders.dataset_severstal import prepare_custom_split
    train_data, val_data = prepare_custom_split(
        train_dir="datasets/severstal/train_inner_split",
        test_dir="datasets/severstal/val_inner_split"
    )

    # Modelo y predictor
    sam2_model = build_sam2(args.config, args.checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    logger = MetricsLogger("logs/training_metrics_2.csv")

    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.2)
    scaler = torch.cuda.amp.GradScaler()

    # Configuración
    NO_OF_STEPS = args.steps
    accumulation_steps = 4
    best_val_iou = 0.0
    mean_iou = 0.0
    subset_name = args.dataset
    MODEL_NAME = args.model_name
    ckpt_dir = "/home/ptp/sam2/models/severstal_updated"
    os.makedirs(ckpt_dir, exist_ok=True)

    for step in range(1, NO_OF_STEPS + 1):
        with torch.cuda.amp.autocast():
            image, mask, input_point, num_masks = read_batch(
                train_data, annotation_format=args.annotation_format, visualize_data=False)

            if image is None or mask is None or num_masks == 0:
                continue

            input_label = np.ones((input_point.shape[0], 1), dtype=np.int32)
            if input_point.size == 0 or input_label.size == 0:
                continue

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True)

            if unnorm_coords is None or labels is None:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None)

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat[-1].unsqueeze(0) for feat in predictor._features["high_res_feats"]]
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
            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()
                scheduler.step()

            # Media móvil del IoU
            batch_mean_iou = np.mean(iou.cpu().detach().numpy())
            mean_iou = mean_iou * 0.99 + 0.01 * batch_mean_iou

            if step % 100 == 0:
                print(f"Step {step}: Accuracy (IoU) = {mean_iou:.4f}")

            # Evaluación en validación interna cada 500 pasos
            if step % 500 == 0:
                val_iou = evaluate_on_validation_set(predictor, val_data, args)
                print(f"[Validation] Step {step}: Mean IoU = {val_iou:.4f}")
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    save_path = os.path.join(ckpt_dir, f"{MODEL_NAME}_best_step{step}_iou{best_val_iou:.2f}.torch")
                    torch.save(predictor.model.state_dict(), save_path)
                    print(f"  New best model saved at step {step} with IoU {best_val_iou:.2f}")

            logger.log(
                mode="train",
                dataset=subset_name,
                steps=step,
                model_name=MODEL_NAME,
                iou=batch_mean_iou,
                iou50=None,
                iou75=None,
                iou95=None
            )
