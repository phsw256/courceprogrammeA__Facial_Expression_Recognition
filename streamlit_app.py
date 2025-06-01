'''
kl:这个文件从头到尾都是我和inline chat合作完成的，虽然包是复制的lab5的，但写一个main文件还是很不容易的
要实现FER2013的识别，模型和头文件写好在model文件夹，现在主要程序要干什么呢
并非只要写main文件
'''

import torch
import torch.nn as nn
import torch.optim as optim
import os
import streamlit as st
import datetime

st.title("courceprogrammeA_Facial_Expression_Recognition")
st.write('#### kl\'s coruce programme.')
st.wrtie('If u use your phone or browser not big enough, click the > above to choose what u want to do with this programme.')
st.write('Unless locally installed, you can't train the models.')

import torchvision
from PIL import Image
import time

from matplotlib import pyplot as plt
from networkx.drawing import draw_planar
from torch.nn.functional import dropout

from models import SimpleMLP, DeepMLP, ResidualMLP, SimpleCNN, MediumCNN, VGGStyleNet, SimpleResNet
from utils import (
    load_FER2013,
    set_seed,
    train_model,
    evaluate_model,
    plot_training_history,
    visualize_model_predictions,
    visualize_conv_filters,
    model_complexity,
    predict
)

save_directory = './ck'
mode=st.sidebar.radio('要做什么',('Train', 'Just_Test', 'Use'))
recorder=[]

if mode=="Train":

    model_type = st.selectbox(
        label = '选择要训练的模型',
        options = ('simple_mlp', 'deep_mlp', 'residual_mlp', 'simple_cnn', 'medium_cnn', 'vgg_style', 'resnet'),
        index = 0,
        format_func = str,
        help = '这里有个问号你可以点击'
    )

    epochs = st.slider(
        label='epochs',
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="其实不用点击，鼠标移上去就行"
    )

    batch_size = st.slider(
        label='batch_size',
        min_value=1,
        max_value=512,
        value=128,
        step=1,
        help="你可以试试512"
    )

    learning_rate = st.number_input(
        label='learning_rate',
        min_value=0.0001,
        max_value=0.1,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="步子迈大了，容易出事"
    )


    use_data_augmentation = st.checkbox(
        label='是否采用数据增强',
        value=True,
        help='''use_data_augmentation = st.checkbox(
        label='是否采用数据增强',
        value=True,
        help=""
    )'''
    )

    visualize_filters = st.checkbox(
        label='是否可视化卷积核',
        value=True,
        help="仅对CNN有效"
    )

    visualize_predictions = st.checkbox(
        label='是否可视化预测结果',
        value=True,
        help="这里不知道填什么了"
    )

    set_seed(st.number_input(
        label='随机种子',
        min_value=0,
        max_value=100,
        value=42,
        step=1,
        help="42是对的"
    ))

    in_channels = None
    if model_type == 'resnet':
        in_channels = st.slider(label='in_channels', value=16, min_value=1, max_value=128, step=1)

    if model_type == 'deep_mlp' or model_type == 'residual_mlp':
        dropout_rate = st.slider(label='dropout_rate', value=0.5, min_value=0.0, max_value=1.0, step=0.01)

    if model_type == 'medium_cnn':
        o1 = st.slider(label='out_channel1', value=32, min_value=8, max_value=128, step=1)

    if model_type == 'vgg_style':
        o2 = st.slider(label='out_channel1', value=64, min_value=8, max_value=256, step=1)

    if model_type == 'residual_mlp':
        activation = st.selectbox(
            label='激活函数',
            options=['relu', 'leaky_relu', 'gelu', 'swish'],
            index=0,
            format_func=str,
        )

    if st.button("Go!"):
        # kl想把超参数记下来
        recorder = [("batchsize", batch_size), ("epochs", epochs), ("learningrate", learning_rate)]

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        st.success(f"使用设备: {device}")

        train_loader, valid_loader, test_loader, classes = load_FER2013(
            use_augmentation=use_data_augmentation,
            batch_size=batch_size
        )

        if model_type == 'simple_mlp':
            model = SimpleMLP()
            model_name = "SimpleMLP"
        elif model_type == 'deep_mlp':
            model = DeepMLP(dropout_rate=dropout_rate, use_bn=True, use_dropout=True)
            recorder.append(("dropout_rate", dropout_rate))
            model_name = "DeepMLP"
        elif model_type == 'residual_mlp':
            model = ResidualMLP(dropout_rate=dropout_rate, activation=activation)
            recorder.append(("dropout_rate", dropout_rate))
            recorder.append(("activation", activation))
            model_name = "ResidualMLP"
        elif model_type == 'simple_cnn':
            model = SimpleCNN()
            model_name = "SimpleCNN"
        elif model_type == 'medium_cnn':
            model = MediumCNN(out_channel1= o1,use_bn=True)
            recorder.append(("out_channel1", o1))
            model_name = "MediumCNN"
        elif model_type == 'vgg_style':
            model = VGGStyleNet(out_channel1= o2)
            recorder.append(("out_channel1", o2))
            model_name = "VGGStyleNet"
        elif model_type == 'resnet':
            model = SimpleResNet(num_blocks=[2, 2, 2],in_channels=in_channels)
            recorder.append(("in_channels", in_channels))
            model_name = "SimpleResNet"
        else:
            st.error("模型类型不匹配，请检查模型文件名。")
            st.stop()

        print(f"使用模型: {model_name}")
        st.success(f"使用模型: {model_name}")



        # 计算模型复杂度
        print("\n分析模型复杂度:")
        model_complexity(model, device=device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 可以添加学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 确保checkpoints目录存在
        os.makedirs(save_directory, exist_ok=True)


        # 训练模型
        trained_model, history = train_model(
            model, train_loader, valid_loader, criterion, optimizer, scheduler,
            num_epochs=epochs, device=device, save_dir=save_directory ,recorder=recorder
        )

        # 绘制训练历史
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_training_history(history, title=f"{timestamp} {model_name} Training History")

        # 在测试集上评估模型
        print("\n在测试集上评估模型:")
        test_loss, test_acc = evaluate_model(trained_model, test_loader, criterion, device, classes ,timestamp)

        st.success(f"{model_name} 最终测试准确率: {test_acc:.4f}")


        # 如果是CNN模型并且需要可视化卷积核
        if visualize_filters and model_type in ['simple_cnn', 'medium_cnn', 'vgg_style', 'resnet']:
            st.write("\n可视化卷积核:")
            if model_type == 'simple_cnn':
                visualize_conv_filters(trained_model, 'conv1',title=f"{timestamp} {model_name} SimpleCNN Conv1 Filters")
            elif model_type == 'medium_cnn':
                visualize_conv_filters(trained_model, 'conv1', title=f"{timestamp} {model_name} MediumCNN Conv1 Filters")
            elif model_type == 'vgg_style':
                visualize_conv_filters(trained_model, 'features.0', title=f"{timestamp} {model_name} VGGStyleNet Conv1 Filters")
            else:  # resnet
                visualize_conv_filters(trained_model, 'conv1', title=f"{timestamp} {model_name} ResNet Conv1 Filters")

        # 如果需要可视化模型预测
        if visualize_predictions:
            st.write("\n可视化模型预测:")
            visualize_model_predictions(trained_model, test_loader, classes, device,title=f"{timestamp} {model_name} Predictions")

        print(f"\n{model_name}的训练和评估已完成！")
        st.success(f"\n{model_name}的训练和评估已完成！")


# 获取 ck 文件夹中的模型文件
model_files = [f for f in os.listdir(save_directory) if f.endswith('.pth')]
# 创建下拉列表供用户选择模型
selected_model = st.selectbox("选择模型文件", model_files)

if mode=="Just_Test":

    if not model_files:
        st.error("ck 文件夹中没有模型文件！")
    else:
        st.success(f"已选择模型文件: {selected_model}")

        use_data_augmentation = st.checkbox(
            label='是否采用数据增强',
            value=True,
            help='''use_data_augmentation = st.checkbox(
                label='是否采用数据增强',
                value=True,
                help=""
            )'''
        )

        visualize_filters = st.checkbox(
            label='是否可视化卷积核',
            value=True,
            help="仅对CNN有效"
        )

        visualize_predictions = st.checkbox(
            label='是否可视化预测结果',
            value=True,
            help="这里不知道填什么了"
        )

        batch_size = st.slider(
            label='batch_size',
            min_value=1,
            max_value=512,
            value=128,
            step=1,
            help="你可以试试512"
        )

        selected_model_name=(selected_model.replace("_best.pth", "")).lower()
        st.success(selected_model_name)

        # 加载模型
        model_path = os.path.join(save_directory, selected_model)
        # 实例化模型对象
        if selected_model_name == 'simplemlp':
            model = SimpleMLP()
        elif selected_model_name == 'deepmlp':
            model = DeepMLP(dropout_rate=0.5, use_bn=True, use_dropout=True)
        elif selected_model_name == 'residualmlp':
            model = ResidualMLP(activation='relu')
        elif selected_model_name == 'simplecnn':
            model = SimpleCNN()
        elif selected_model_name == 'mediumcnn':
            model = MediumCNN(use_bn=True)
        elif selected_model_name == 'vggstylenet':
            model = VGGStyleNet()
        elif selected_model_name == 'simpleresnet':  # resnet
            model = SimpleResNet(num_blocks=[2, 2, 2])
        else:
            st.error("模型类型不匹配，请检查模型文件名。")
            st.stop()

        # 加载模型参数
        model.load_state_dict(torch.load(model_path))

        st.success(f"模型 {selected_model} 已加载成功！")

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        st.success(f"使用设备: {device}")

    if st.button("Go!"):
        train_loader, valid_loader, test_loader, classes = load_FER2013(
            use_augmentation=use_data_augmentation,
            batch_size=batch_size
        )

        criterion = nn.CrossEntropyLoss()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n在测试集上评估模型:")
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, classes, timestamp)

        st.success(f"{selected_model} 最终测试准确率: {test_acc:.4f}")

        # 如果是CNN模型并且需要可视化卷积核
        if visualize_filters and selected_model_name in ['simple_cnn', 'medium_cnn', 'vgg_style', 'resnet']:
            print("\n可视化卷积核:")
            if selected_model_name == 'simple_cnn':
                visualize_conv_filters(model, 'conv1', title=f"{timestamp} {selected_model} SimpleCNN Conv1 Filters")
            elif selected_model_name == 'medium_cnn':
                visualize_conv_filters(model, 'conv1', title=f"{timestamp} {selected_model} MediumCNN Conv1 Filters")
            elif selected_model_name == 'vgg_style':
                visualize_conv_filters(model, 'features.0',
                                       title=f"{timestamp} {selected_model} VGGStyleNet Conv1 Filters")
            else:  # resnet
                visualize_conv_filters(model, 'conv1', title=f"{timestamp} {selected_model} ResNet Conv1 Filters")

        # 如果需要可视化模型预测
        if visualize_predictions:
            print("\n可视化模型预测:")
            visualize_model_predictions(model, test_loader, classes, device,
                                        title=f"{timestamp} {selected_model} Predictions")

        print(f"\n{selected_model}的评估已完成！")
        st.success(f"\n{selected_model}的评估已完成！")


if mode=="Use":

    if not model_files:
        st.error("ck 文件夹中没有模型文件！")
    else:
        st.success(f"已选择模型文件: {selected_model}")

        selected_model_name = (selected_model.replace("_best.pth", "")).lower()
        st.success(selected_model_name)

        # 加载模型
        model_path = os.path.join(save_directory, selected_model)
        # 实例化模型对象
        if selected_model_name == 'simplemlp':
            model = SimpleMLP()
        elif selected_model_name == 'deepmlp':
            model = DeepMLP(dropout_rate=0.5, use_bn=True, use_dropout=True)
        elif selected_model_name == 'residualmlp':
            model = ResidualMLP(activation='relu')
        elif selected_model_name == 'simplecnn':
            model = SimpleCNN()
        elif selected_model_name == 'mediumcnn':
            model = MediumCNN(use_bn=True)
        elif selected_model_name == 'vggstylenet':
            model = VGGStyleNet()
        elif selected_model_name == 'simpleresnet':
            model = SimpleResNet(num_blocks=[2, 2, 2])
        else:
            st.error("模型类型不匹配，请检查模型文件名。")
            st.stop()

        # 加载模型参数
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

        st.success(f"模型 {selected_model} 已加载成功！")

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        st.success(f"使用设备: {device}")

        img_source = st.sidebar.radio('Please select the source of the facial expression image.',
                                      ('Upload the image', 'Capture the image', 'Select a sample image', 'Clipboard'))

        if img_source == 'Upload the image':
            img_file = st.sidebar.file_uploader('Please upload the facial expression image.',
                                                type=['jpg', 'png', 'jpeg'])
            if img_file is None:
                st.write('#### ← You can select how to upload the image from the sidebar.')

        elif img_source == 'Capture the image':
            img_file = st.sidebar.camera_input('Please capture the facial expression image.')
            if img_file is None:
                st.write('#### ← You can select how to upload the image from the sidebar.')
        elif img_source == 'Select a sample image':
            img_file = st.sidebar.radio(
                'Please select a sample image.',
                ('Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6')
            )

            if img_file == 'Sample 1':
                img_file = 'image/sample1.jpg'
            elif img_file == 'Sample 2':
                img_file = 'image/sample2.jpg'
            elif img_file == 'Sample 3':
                img_file = 'image/sample3.jpg'
            elif img_file == 'Sample 4':
                img_file = 'image/sample4.jpg'
            elif img_file == 'Sample 5':
                img_file = 'image/sample5.jpg'
            elif img_file == 'Sample 6':
                img_file = 'image/sample6.jpg'
            else:
                img_file = None
        else:#img_source == 'Clipboard':
            from PIL import ImageGrab
            img_file = None
            try:
                img = ImageGrab.grabclipboard()
                if img is None:
                    st.error("剪贴板中没有图片！")
                else:
                    img_file = img.convert('RGB')
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    os.makedirs('image', exist_ok=True)  # 确保文件夹存在
                    img_file.save(f'image/clipboard_image_{timestamp}.png')
                    img_file = f'image/clipboard_image_{timestamp}.png'
            except Exception as e:
                st.error(f"无法从剪贴板读取图片: {e}")

        if img_file is not None:
            if st.button("Go!"):
                with st.spinner('loading・・・'):
                    start_time = time.time()
                    img = Image.open(img_file)

                    transform = torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(num_output_channels=3),
                        torchvision.transforms.Resize((48, 48)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

                    transimg_tensor = transform(img)
                    transimg_np = transimg_tensor.permute(1, 2, 0).numpy()

                    st.image(img, caption='Facial expression image', use_column_width=True)

                    results = predict(transimg_tensor, model, device=device)
                    st.success(f'Elapsed time: {time.time() - start_time:.3f} [sec]')
                    st.subheader('Probs of each emoji:')

                    # Display bar chart
                    st.bar_chart(data=results)

                    # Display emotion
                    emotion = results.idxmax()[0]
                    st.subheader('Predicted emoji:')

                    # Display Big Emoji
                    st.markdown(f'<h1 style="text-align:center;">{emotion}</h1>', unsafe_allow_html=True)

                    # 生成PNG图片报告
                    report_path = f"./reportfig/report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{selected_model}.png"
                    plt.figure(figsize=(8, 24))
                    plt.suptitle(f"Expression Analysis Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", fontsize=16)
                    plt.subplot(3, 1, 1)
                    plt.title("input image")
                    plt.imshow(img)
                    plt.axis('off')
                    plt.subplot(3, 1, 2)
                    plt.title("transformed image")
                    plt.imshow((transimg_np * 0.5 + 0.5))
                    plt.axis('off')
                    plt.subplot(3, 1, 3)
                    plt.title(f'Probabilities__model:{selected_model}')
                    plt.bar(results.index, results['Probability'], tick_label=results.index)
                    plt.xlabel("Emoji")
                    plt.ylabel("Probability")
                    plt.axis('on')
                    os.makedirs('./reportfig', exist_ok=True)  # 确保文件夹存在
                    plt.savefig(report_path)
                    st.pyplot(plt.gcf())  # 显示图像
                    plt.close()

                    st.success(f"报告已生成并保存在 {report_path}")



        st.sidebar.divider()

        st.sidebar.caption('This app is not powered by Hugging Face 🤗 .  \n \
                                The ViT model was not fine-tuned on [FER2013 dataset](https://paperswithcode.com/dataset/fer2013).\n \
                                But we have lab5.')
