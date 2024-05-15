import cv2

# 读取图像
image = cv2.imread("/Users/oppressor/Downloads/mouse_phone_headphone.JPG")

if image is None:
    print("无法加载图像，请检查文件路径或图像格式是否正确")
else:
    # 设置目标宽度和高度
    target_width = 640
    target_height = 480

    # 缩小图像
    resized_image = cv2.resize(image, (target_width, target_height))

    # 保存缩小后的图像
    cv2.imwrite('/Users/oppressor/Downloads/output_image.jpg', resized_image)