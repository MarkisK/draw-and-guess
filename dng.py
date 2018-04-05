from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    # if request.method == 'POST':
    #     code to post
    return render_template('index.html')

if __name__ == "__main__":
    # disable debug mode for production
    app.run(debug=True)
