from flask import Flask

app = Flask(__name__)

@app.route('/hello/<name>')
def Homepage(name):
    return 'Hello %s how are You !' % name

@app.route('/blog/<int:postID>')
def show_blog(postID):
    return 'YOUR BLOG NUMBER %d '% postID

@app.route('/rev/<float:revNO>')
def revision(revNO):
    return 'Revision Number %f '% revNO

if __name__ == '__main__':
    app.run(debug=True)