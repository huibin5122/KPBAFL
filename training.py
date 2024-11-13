import argparse
import shutil
from datetime import datetime
import yaml
from copy import deepcopy
from prompt_toolkit import prompt
from tqdm import tqdm
from helper import Helper
from utils.utils import *
logger = logging.getLogger('logger')

def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True, global_model=None):
    losses = []  # 创建一个空列表来存储损失值

    criterion = hlpr.task.criterion
    model.train()
    for i, data in enumerate(train_loader):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, global_model)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i == hlpr.params.max_batch_id:
            break
    print('hb epoch：',epoch,'loss:', losses)
    return

def test(hlpr: Helper, epoch, backdoor=False, model=None):
    if model is None:
        model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()

    #hb add
    total_loss = 0.0  # 用于记录总损失值
    correct_predictions = 0  # 用于记录正确预测的样本数
    num_samples = 0  # 用于记录总样本数
    #hb end

    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader), desc='测试', leave=False):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            loss = hlpr.task.criterion(outputs, batch.labels)  # 计算损失值
            total_loss += loss.mean().item() * batch.inputs.size(0)  # 将批次损失值乘以批次大小并累加
            num_samples += batch.inputs.size(0)  # 累加样本数量

            # 预测并统计正确预测的样本数
            predictions = outputs.argmax(dim=1)  # 获取预测结果
            correct_predictions += (predictions == batch.labels).sum().item()

            print(f'批次： {i} ，损失值: {loss.mean().item():.4f}，')  # 输出平均损失值

            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                              prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    accuracy = correct_predictions / num_samples  # 计算准确率
    avg_loss = total_loss / num_samples  # 计算平均损失值
    print(f'平均损失值: {avg_loss:.4f}')  # 输出平均损失值
    print(f'平均损失值: {accuracy:.4f}，')  # 输出平均损失值
    return metric

def run_fl_round(hlpr: Helper, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model
    local_model_hb=hlpr.task.local_model
    round_participants = hlpr.task.sample_users_for_round(epoch) #?根据轮次为用户分配数据，并判断是否为攻击者
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in tqdm(round_participants,desc="用户进程", leave=False):
        hlpr.task.copy_params(global_model, local_model)#将global复制到local
        hlpr.task.copy_params(global_model, local_model_hb)
        optimizer = hlpr.task.make_optimizer(local_model)
        optimizer_hb = hlpr.task.make_optimizer(local_model_hb)
        if user.compromised:
            if not user.user_id == 0:
                continue
            for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs),desc=f"用户 {user.user_id} 投毒训练", leave=False):
                #print(f'投毒，当前为{local_epoch}/{hlpr.params.fl_poison_epochs}')
                #### local_model 中毒模型
                #### local_model_hb  正常模型
                train(hlpr, local_epoch, local_model, optimizer, user.train_loader, attack=True, global_model=global_model)  #训练投毒模型
                #train(hlpr, local_epoch, local_model_hb, optimizer_hb, user.train_loader, attack=False, global_model=global_model) #训练正常模型

            tqdm(range(hlpr.params.fl_poison_epochs), desc=f"用户 {user.user_id} 投毒训练", leave=False).close()
            hlpr.save_update_model(model=local_model,Attack=True)
            #hlpr.save_update_model(model=local_model_hb, userID=user.user_id, Attack=False)
            #local_update_hb = hlpr.attack.get_fl_update(local_model_hb, global_model)
            # 使用示例
            #compare_models(local_model, local_model_hb)
            local_update = hlpr.attack.get_fl_update(local_model, global_model)  # 计算本地模型与全局模型之间的参数更新量
            hlpr.save_update(model=local_update, userID=user.user_id)

        else:
            if user.user_id==50:
                for local_epoch in tqdm(range(hlpr.params.fl_local_epochs), desc=f"用户 {user.user_id} 正常训练",leave=False):
                    train(hlpr, local_epoch, local_model, optimizer, user.train_loader, attack=False)
                hlpr.save_update_model(model=local_model, Attack=False)
            else:
                for local_epoch in tqdm(range(hlpr.params.fl_local_epochs),desc=f"用户 {user.user_id} 正常训练",leave=False):

                    train(hlpr, local_epoch, local_model, optimizer, user.train_loader, attack=False)
            tqdm(range(hlpr.params.fl_local_epochs), desc=f"用户 {user.user_id} 正常训练", leave=False).close()
            local_update = hlpr.attack.get_fl_update(local_model, global_model)  # 计算本地模型与全局模型之间的参数更新量
            hlpr.save_update(model=local_update, userID=user.user_id)

        if user.compromised:
            hlpr.attack.local_dataset = deepcopy(user.train_loader)


    hlpr.attack.perform_attack(global_model, epoch)  #基础是更新所有恶意客户端本地泉州
    hlpr.defense.aggr(weight_accumulator, global_model)
    hlpr.task.update_global_model(weight_accumulator, global_model)


# def compare_models(local_model, local_model_hb, middle_ratio=0.8):
#     # 计算参数差异
#     param_diff = {}
#     for name, param in local_model.named_parameters():
#         param_diff[name] = torch.norm(param - local_model_hb.state_dict()[name])
#
#     # 按参数差异排序
#     sorted_params = sorted(param_diff.items(), key=lambda x: x[1], reverse=True)
#
#     # 计算中间80%的范围
#     total_params = len(param_diff)
#     start_index = int(total_params * (1 - middle_ratio) / 2)
#     end_index = int(total_params * (1 + middle_ratio) / 2)
#
#     # 选择中间80%的参数
#     selected_params = sorted_params[start_index:end_index]
#
#     # 替换参数
#     for name, _ in selected_params:
#         local_model.state_dict()[name].data.copy_(local_model_hb.state_dict()[name].data)





def run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)  #d度量
        #print(f'main acc:{metric}，Epoch:{epoch}')
        back_acc=test(hlpr, epoch, backdoor=True)
        #print(f'backdoor acc:{back_acc}，Epoch:{epoch}')
        hlpr.record_accuracy(metric, back_acc, epoch)

        hlpr.save_model(hlpr.task.model, epoch, metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', required=True)
    parser.add_argument('--name', dest='name', required=True)
    # parser.add_argument('--hb_normal', dest='hb_normal', required=False)
    # parser.add_argument('--hb_poision', dest='hb_poision', required=False)
    # parser.add_argument('--cs_constraint', dest='cs_constraint', required=False)
    # parser.add_argument('--eu_constraint', dest='eu_constraint', required=False)
    # parser.add_argument('--hb_replacement_percent', dest='hb_replacement_percent', required=False)

    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name

    helper = Helper(params)
    logger.warning(create_table(params))

    try:
        run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. ")
        else:
            logger.error(f"Aborted training. No output generated.")
    helper.remove_update()