import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 加载模型
model = tf.keras.models.load_model('F:\s1\pythonProject2\saved_model\cat_dog_classifier_model.h5')


# 预测函数
def predict_image(img_path):
    # 加载图像并调整为模型所需的尺寸
    img = image.load_img(img_path, target_size=(150, 150))

    # 将图像转换为 numpy 数组
    img_array = image.img_to_array(img)

    # 扩展维度，将图像形状从 (150, 150, 3) 转换为 (1, 150, 150, 3)，以适应模型输入
    img_array = np.expand_dims(img_array, axis=0)

    # 归一化处理
    img_array /= 255.0

    # 进行预测
    prediction = model.predict(img_array)

    # 输出预测结果
    if prediction[0] > 0.5:
        print(f"{img_path} 是狗 (概率: {prediction[0][0]:.2f})")
    else:
        print(f"{img_path} 是猫 (概率: {1 - prediction[0][0]:.2f})")


# 输入图像路径
img_path = 'F:\dataset\PetImages\Test\Cat\Doooo.jpg'  # 替换为你想预测的图像的路径
predict_image(img_path)
