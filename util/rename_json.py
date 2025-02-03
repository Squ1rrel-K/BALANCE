import json


def rename_json():
    # 读取 JSON 文件
    input_file = "../config/train.json"  # 输入文件名
    output_file = "../config/train.json"  # 输出文件名

    # 从文件加载 JSON 数据
    with open(input_file, 'r') as f:
        data = json.load(f)

    byzantine_attack = data["byzantine_settings"]["byzantine_attack"]
    byzantine_defend = data["byzantine_settings"]["byzantine_defend"]
    adv_rate = data["byzantine_settings"]["adv_rate"] * 100
    epochs = data["fl_settings"]["epochs"]
    client_size = data["fl_settings"]["client_size"]
    dataset = data["fl_settings"]["dataset"]
    model = data["fl_settings"]["model"]

    # 打印所有参数
    print("Byzantine Attack:", byzantine_attack)
    print("Byzantine Defend:", byzantine_defend)
    print("Adversarial Rate:", adv_rate, "%")
    print("Epochs:", epochs)
    print("Client Size:", client_size)
    print("Dataset:", dataset)
    print("Model:", model)

    # 拼接新名字，并打印出来
    new_report_name = f"{model}_{dataset}_{byzantine_attack}_{byzantine_defend}_adv_{adv_rate}_epochs_{epochs}_clients_{client_size}"
    print("New Report Name:", new_report_name)

    # 修改 report_name 的值
    data["other_settings"]["report_name"] = new_report_name

    # 将修改后的 JSON 数据写回文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    # 打印确认信息
    print(f"修改后的 JSON 数据已保存到 {output_file}")


# if __name__ == '__main__':
#     rename_json()