import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from network import pvt_cls as TCN
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Predict Only')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--batch-size', type=int, default=8, help='batch size for prediction')
parser.add_argument('--crop-size', type=int, default=256, help='the crop size of the test image')
parser.add_argument('--model-path', type=str, required=True, help='saved model path')
parser.add_argument('--data-path', type=str, required=True, help='dataset path')

def predict(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    dataset = crowd.Crowd_TC(os.path.join(args.data_path, 'test_data'), args.crop_size, 1, method='val')
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)
        
    model = TCN.pvt_treeformer(pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, device))
    model.eval()

    for inputs, _, name, _ in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            crop_imgs, crop_masks = [], []
            b, c, h, w = inputs.size()
            rh, rw = args.crop_size, args.crop_size
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros([b, 1, h, w]).to(device)
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))
            crop_preds = []
            nz, bz = crop_imgs.size(0), args.batch_size
            for i in range(0, nz, bz):
                gs, gt = i, min(nz, i + bz)
                crop_pred, _ = model(crop_imgs[gs:gt])
                crop_pred = crop_pred[0]
                _, _, h1, w1 = crop_pred.size()
                crop_pred = F.interpolate(crop_pred, size=(h1 * 4, w1 * 4), mode='bilinear', align_corners=True) / 16
                crop_preds.append(crop_pred)
            crop_preds = torch.cat(crop_preds, dim=0)
            idx = 0
            pred_map = torch.zeros([b, 1, h, w]).to(device)
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                    idx += 1
            mask = crop_masks.sum(dim=0).unsqueeze(0)
            outputs = pred_map / mask
            pred_count = torch.sum(outputs).item()
            print(f"Image: {name[0]}, Predicted count: {pred_count}")

if __name__ == '__main__':
    args = parser.parse_args()
    predict(args)