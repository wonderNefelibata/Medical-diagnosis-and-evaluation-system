from flask import Flask, render_template, request, jsonify
import os
from part1.api import diagnosis, transform_image
from part2.api import sentiment_classify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict_pneumonia', methods=['POST'])
def predict_pneumonia():
    # 获取上传的图片文件
    file = request.files['file']

    # 保存图片到临时文件
    img_path = os.path.join('./temp', file.filename)
    file.save(img_path)

    # 对图片进行预处理
    img = transform_image(img_path)

    # 调用诊断函数
    result = diagnosis(img)

    if result == 0:
        diagnosis_result = "Covid"
    elif result == 1:
        diagnosis_result = "Normal"
    else:
        diagnosis_result = "Viral Pneumonia"
    print(diagnosis_result)
    # 返回诊断结果
    return jsonify({'result': diagnosis_result})


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    # 检查请求头是否是multipart/form-data
    if 'review' not in request.form:
        return jsonify({"error": "No review text provided."}), 400
    print("111:")
    review = request.form['review']
    print("review:"+review)
    # 调用API进行判断
    result = sentiment_classify(review)
    print(result)
    if result == 0:
        predict_res = "很差"
    elif result == 1:
        predict_res = "中等"
    else:
        predict_res = "很好"
    print(predict_res)
    return jsonify({'result': predict_res})


if __name__ == '__main__':
    app.run(debug=True)
