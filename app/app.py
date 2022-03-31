from flask import Flask
import tela

app = Flask(__name__)
app_dash = tela.init_app(app)
app = app_dash.server

@app.route("/")
def my_dash_app():
    return app_dash.index()

if __name__=='__main__':
    app_dash.run_server(host='0.0.0.0',port=8050,debug=False)