from flask import Flask, render_template
from flask_socketio import SocketIO


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('websocket.html')

@socketio.on('update string')
def handle_update_string(data):
    # Update the string with the data sent from the client
    # You can use the emit() function to send the updated string to the client
    i = 0
    while True:
        i = i + 1
        updated_string = str(i)
        socketio.emit('string updated', {'string': updated_string})


if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True)
