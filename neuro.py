import torch
from attacks.attack import Attack
from attacks.components.decoy import decoy_model_design, benign_training
import logging
import numpy as np

logger = logging.getLogger('logger')
class neuro(Attack):

    def __init__(self, params, synthesizer):
        super().__init__(params, synthesizer)

        self.loss_tasks.append('eu_constraint')
        self.loss_tasks.append('cs_constraint')
        # self.fixed_scales = {'normal': 0.2,
        #                      'backdoor': 0.2,
        #                      'eu_constraint': 0.6}
        self.fixed_scales = {'normal': params.hb_normal,
                             'backdoor': params.hb_poision,
                             'eu_constraint': params.eu_constraint,
                             'cs_constraint':params.cs_constraint}


    def perform_attack(self, global_model, epoch):
        if self.params.fl_number_of_adversaries <= 0 or \
                epoch not in range(self.params.poison_epoch, self.params.poison_epoch_stop):
            return

        folder_name = f'{self.params.folder_path}/saved_updates'
        file_name = f'{folder_name}/update_0.pth'

        # file_name_begin=f'{folder_name}/update_0_beginmodel.pth'
        # file_name_back = f'{folder_name}/update_0_backmodel.pth'

        #mnist 改成10
        #cifar 改成30
        # file_name_nomol = f'{folder_name}/update_10.pth'
        # nomol_update=torch.load(file_name_nomol)
        #
        # begin_model = torch.load(file_name_begin)
        # back_model = torch.load(file_name_back)

        #模型参数打印
        #print_model_parameters(back_model)

        #region 计算后门模型与良性模型之间每个参数的差异并保存

        # backdoor_params = dict(back_model.named_parameters())
        # benign_params = dict(begin_model.named_parameters())
        # param_diff1 = {}
        # bb=0
        # for name in backdoor_params.keys():
        #     backdoor_param = backdoor_params[name].detach().cpu().numpy()  # 将张量从 GPU 移动到 CPU
        #     benign_param = benign_params[name].detach().cpu().numpy()  # 将张量从 GPU 移动到 CPU
        #     diff = backdoor_param - benign_param
        #     param_diff1[name] = diff
        #     bb=bb+1
        # sorted_param_diff = sorted(param_diff1.items(), key=lambda x: np.linalg.norm(x[1]), reverse=True)
        # print(f"层数：{bb}")
        # with open(f'{folder_name}/aaa.txt', "w") as f:
        #     for name, diff in sorted_param_diff:
        #         f.write(f"Parameter0: {name}, Difference: {np.linalg.norm(diff)}\n")
        #         print(f"Parameter0: {name}, Difference: {np.linalg.norm(diff)}")
        #endregion

        #compare_models(back_model, begin_model)
        #backdoor_update = self.get_fl_update(back_model, global_model)

        #self.scale_update(backdoor_update, self.params.fl_weight_scale)  # 模型权重增加

        # Benign training
        # benign_model = benign_training(self.params, global_model, self)
        # benign11_params = dict(benign_model.named_parameters())
        # param_diff2 = {}
        # aa=0
        # for name in backdoor_params.keys():
        #     backdoor_param = backdoor_params[name].detach().cpu().numpy()  # 将张量从 GPU 移动到 CPU
        #     benign_param = benign11_params[name].detach().cpu().numpy()  # 将张量从 GPU 移动到 CPU
        #     diff = backdoor_param - benign_param
        #     param_diff2[name] = diff
        #     aa=aa+1
        # sorted_param_diff = sorted(param_diff2.items(), key=lambda x: np.linalg.norm(x[1]), reverse=True)
        # print(f'层数:{aa}')
        # with open(f'{folder_name}/bbb.txt', "w") as f:
        #     for name, diff in sorted_param_diff:
        #         f.write(f"Parameter1: {name}, Difference: {np.linalg.norm(diff)}\n")
        #         print(f"Parameter1: {name}, Difference: {np.linalg.norm(diff)}")



        benign_model = benign_training(self.params, global_model, self)
        benign_update = self.get_fl_update(benign_model, global_model)  # 返回更新量
        benign_norm = self.get_update_norm(benign_update)  # 计算范数






        # If the norm is so small, scale the norm to the magnitude of benign reference update 如果规范很小，则将规范缩放到良性参考更新的大小
        backdoor_update = torch.load(file_name)

        #操作
        backdoor_update= replace_params(backdoor_update, benign_update, replacement_percent=self.params.hb_replacement_percent)

        backdoor_norm = self.get_update_norm(backdoor_update)

        scale_f = min((benign_norm / backdoor_norm), self.params.fl_weight_scale)
        logger.info(f'neuro: 差距{benign_norm / backdoor_norm}')
        logger.info(f"neuro: scaling factor is {max(scale_f, 1)}")

        self.scale_update(backdoor_update, max(scale_f, 1))

        # print("backupdate参数数量：", count_parameters_in_update(backdoor_update))
        # print("nomalupdate参数数量：", count_parameters_in_update(nomol_update))
        # num_layers0= count_layers(back_model)
        # num_layers1 = count_layers(begin_model)
        # num_layers2 = count_layers(benign_model)
        # print(f"backmodel模型共有 {num_layers0} 层")
        # print(f"beginmodel模型共有 {num_layers1} 层")
        # print(f"全局模型共有 {num_layers2} 层")

        # print("backmodel参数：", count_parameters_in_update(back_model))
        # print("bbegin参数：", count_parameters_in_update(begin_model))

        for i in range(self.params.fl_number_of_adversaries):
            file_name = f'{folder_name}/update_{i}.pth'
            torch.save(backdoor_update, file_name)



def replace_params(backdoor_update, benign_update, replacement_percent=0.012):
    # 从字典中提取参数并保持在 GPU 上
    backdoor_params = torch.cat([param.flatten() for param in backdoor_update.values()]).detach()
    benign_params = torch.cat([param.flatten() for param in benign_update.values()]).detach()

    # 将参数转移到 CPU 上进行排序
    backdoor_params_cpu = backdoor_params.cpu().numpy()
    benign_params_cpu = benign_params.cpu().numpy()

    # 计算要替换的参数数量
    num_replace = int(replacement_percent * len(backdoor_params_cpu))
    aa=len(backdoor_params_cpu)
    print(f'参数总量{aa}')
    print(f'要替换的参数数量：{num_replace}')
    logger.info(f'can shu zong linag {aa}')
    logger.info(f'要替换的参数数量 {num_replace}')
    # 通过排序绝对差值找到要替换的参数的索引
    indices_to_replace = np.argsort(np.abs(backdoor_params_cpu - benign_params_cpu))[:num_replace]

    # 用benign_update中的参数替换backdoor_update中的参数
    for idx in indices_to_replace:
        param_idx = 0
        for param in backdoor_update.values():
            param_size = np.prod(param.shape)
            if idx < param_size:
                param = param.flatten()
                param[idx] = benign_params[idx]
                param = param.reshape(param.shape).to('cuda:0')  # 转回Tensor并放到 GPU 上
                break
            else:
                idx -= param_size

    return backdoor_update

'''
参数替换
percentage 前多少
替换后门模型中的一些参数
'''
# def replace_parameters(normal_model, backdoor_model, percentage=0.2):
#     # 将后门模型的参数移动到与正常模型相同的 GPU 上
#     device = next(normal_model.parameters()).device
#     backdoor_model.to(device)
#
#     # 获取正常模型和后门模型的参数
#     normal_params = [param for param in normal_model.parameters()]
#     backdoor_params = [param for param in backdoor_model.parameters()]
#
#     # 将参数展平并按大小排序
#     normal_flat_params = torch.cat([param.flatten() for param in normal_params])
#     sorted_indices = torch.argsort(normal_flat_params, descending=True)
#
#     # 获取要替换的参数的数量
#     num_params_to_replace = int(len(sorted_indices) * percentage)
#
#     # 计算总参数数量
#     total_params = sum(param.numel() for param in normal_params)
#
#     # 将 normal_flat_params 移动到与 backdoor_params 相同的设备上
#     normal_flat_params = normal_flat_params.to(backdoor_params[0].device)
#
#     # 将正常模型的参数值替换后门模型的对应参数值，并输出差异值
#     replaced_params_count = 0
#     for idx in sorted_indices[:num_params_to_replace]:
#         layer_index, flat_index = 0, idx
#         for param in backdoor_params:
#             param_size = torch.prod(torch.tensor(param.size()))
#             if replaced_params_count + param_size > idx:
#                 break
#             replaced_params_count += param_size
#             layer_index += 1
#
#         normal_value = normal_flat_params[idx].item()
#         backdoor_value = backdoor_params[layer_index].flatten()[idx - replaced_params_count].item()
#         difference = normal_value - backdoor_value
#         print(f"参数位置: {idx}, 正常模型值: {normal_value}, 后门模型值: {backdoor_value}, 差异值: {difference}")
#
#         backdoor_params[layer_index].flatten()[idx - replaced_params_count] = normal_flat_params[idx]

def replace_backdoor_params(backdoor_model, normal_model, replace_percentage=0.005):
    # 比较参数
    backdoor_params = dict(backdoor_model.named_parameters())
    normal_params = dict(normal_model.named_parameters())

    # 找到要替换的参数数量
    total_params = sum(p.numel() for p in backdoor_model.parameters())
    num_to_replace = int(total_params * replace_percentage)

    # 对参数按照某种规则排序，这里假设按照参数值的大小排序
    sorted_params = sorted(backdoor_params.items(), key=lambda x: torch.abs(x[1].data).sum(), reverse=True)

    # 替换参数
    for name, param in sorted_params[:num_to_replace]:
        param.data.copy_(normal_params[name].data)

    """
    打印模型的所有参数及其值
    """
def print_model_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print('模型参数数量:', num_params)
        print('模型打印', name, param.data)
    print('总参数数量:', total_params)

# 假设 `model` 是已定义好的PyTorch模型实例
def count_layers(model):
    num_layers = 0
    for _, module in model.named_modules():
        # 不计入子模块内的参数，只计数模块（层）
        num_layers += 1
    return num_layers

def count_parameters_in_update(local_update):
    total_params = 0
    for name, value in local_update.items():
        total_params += value.numel()
    return total_params

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_models(local_model, local_model_hb, middle_ratio=0.99):
    # 计算参数差异
    param_diff = {}
    for name, param in local_model.named_parameters():
        param_diff[name] = torch.norm(param - local_model_hb.state_dict()[name])

    # 按参数差异排序
    sorted_params = sorted(param_diff.items(), key=lambda x: x[1], reverse=True)

    # 计算中间80%的范围
    total_params = len(param_diff)
    start_index = int(total_params * (1 - middle_ratio) / 2)
    end_index = int(total_params * (1 + middle_ratio) / 2)

    # 选择中间80%的参数
    selected_params = sorted_params[start_index:end_index]

    # 替换参数
    for name, _ in selected_params:
        local_model.state_dict()[name].data.copy_(local_model_hb.state_dict()[name].data)