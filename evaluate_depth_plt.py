import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff, multiclass_IoU
from torchmetrics import JaccardIndex
from torchmetrics.functional import mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error, mean_squared_error


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = max(1, len(dataloader))

    # calculate AbsRel, SqRel, RMSE, RMSE_log, and a1, a2, a3
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask
            depth_pred = net(image)

            # assume classes = 2
            sq_rel += mean_absolute_percentage_error(depth_pred, mask_true)
            abs_rel += mean_squared_error(depth_pred, mask_true)
            rmse += mean_squared_error(depth_pred, mask_true)
            rmse_log += mean_squared_log_error(depth_pred, mask_true)
            # a1 += multiclass_dice_coeff(depth_pred, mask_true)
            # a2 += multiclass_IoU(depth_pred, mask_true)
            # a3 += multiclass_dice_coeff(depth_pred, mask_true)
        
        abs_rel /= num_val_batches
        sq_rel /= num_val_batches
        rmse /= num_val_batches
        rmse_log /= num_val_batches
        a1 /= num_val_batches
        a2 /= num_val_batches
        a3 /= num_val_batches
    net.train()
    
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3



# function for calc MEANABSOLUTERELATIVEERROR using torch metrics ignit


# AbsRel = np.mean(np.abs(pred_test_depth3-test_depth3)/test_depth3)
# RMSE = np.sqrt(np.mean(np.abs(pred_test_depth3-test_depth3)**2))
# RMSE_log = np.sqrt(np.mean(np.abs(np.log10(pred_test_depth3)-np.log10(test_depth3))**2))
# SqRel = np.mean(np.abs(pred_test_depth3-test_depth3)**2/test_depth3)

# print(f' AbsRel: {AbsRel} \n RMSE: {RMSE} \n RMSE_log: {RMSE_log} \n SqRel: {SqRel} ')



# xnp = pred_test_depth3.reshape(100,480000)
# ynp = test_depth3.reshape(100,480000)     

# #%%
# thr = np.zeros(3)
# acc = np.zeros(3)
# for i in range(100):
#   for j in range(480000):
#       thr[0] += np.max([xnp[i][j]/ynp[i][j],ynp[i][j]/xnp[i][j]]) <1.25
#       thr[1] += np.max([xnp[i][j]/ynp[i][j],ynp[i][j]/xnp[i][j]]) <1.25**2
#       thr[2] += np.max([xnp[i][j]/ynp[i][j],ynp[i][j]/xnp[i][j]]) <1.25**3
#   acc += thr/480000
#   thr = np.zeros(3)

# print(f' Accuracy with threshold: {acc/(i+1)}')
