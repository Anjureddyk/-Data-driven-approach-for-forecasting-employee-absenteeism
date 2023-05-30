import dash
from dash import dcc, html, Input, Output
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the pickled model
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

class_names = ['abyssinian', 'american shorthair', 'beagle', 'boxer', 'bulldog', 'chihuahua', 'corgi', 'dachshund',
               'german shepherd', 'golden retriever', 'husky', 'labrador', 'maine coon', 'mumbai cat', 'persian cat',
               'pomeranian', 'pug', 'ragdoll cat', 'rottwiler', 'shiba inu', 'siamese cat', 'sphynx', 'yorkshire terrier']


# Initialize the Dash application
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Pet Breed Classification'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Image')
        ]),
        style={
            'width': '300px',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-container', style={'margin': '20px'})
])

# Define the callback function
@app.callback(Output('output-container', 'children'),
              [Input('upload-image', 'contents')])
def classify_image(contents):
    if contents is not None:
        # Decode and preprocess the image
        _, content_string = contents.split(',')
        img = image.load_img('C:/Users/Anju Reddy K/Personal_projects/pet breed classification/' + content_string, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make the prediction
        prediction = model.predict(img_array)
        breed_index = np.argmax(prediction)
        predicted_breed = class_names[breed_index]

        # Display the predicted breed
        return html.H2(f'Predicted Breed: {predicted_breed}')

    return html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)
