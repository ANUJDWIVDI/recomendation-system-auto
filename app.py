from flask import Flask, render_template, request
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, send_file, request, redirect
import os
import zipfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image

from flask import Flask, render_template, jsonify

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

products = {
    'Books': [
        {'name': 'Books', 'tags': ['tag1', 'tag2', 'tag3']},
        {'name': 'Book2', 'tags': ['tag2', 'tag3', 'tag4']},
        {'name': 'Book3', 'tags': ['tag1', 'tag4', 'tag5']}
    ],
    'Sports': [
        {'name': 'Sports Product 1', 'tags': ['tag6', 'tag7', 'tag8']},
        {'name': 'Sports Product 2', 'tags': ['tag7', 'tag9', 'tag10']},
        {'name': 'Sports Product 3', 'tags': ['tag6', 'tag10', 'tag11']}
    ],
    'Clothes': [
        {'name': 'Clothing 1', 'tags': ['tag12', 'tag13', 'tag14']},
        {'name': 'Clothing 2', 'tags': ['tag13', 'tag15', 'tag16']},
        {'name': 'Clothing 3', 'tags': ['tag12', 'tag17', 'tag18']}
    ],
    'Gadgets': [
        {'name': 'Gadget 1', 'tags': ['tag19', 'tag20', 'tag21']},
        {'name': 'Gadget 2', 'tags': ['tag19', 'tag22', 'tag23']},
        {'name': 'Gadget 3', 'tags': ['tag20', 'tag23', 'tag24']}
    ]
}



@app.route('/', methods=['GET', 'POST'])
def index12():
    if request.method == 'POST':
        # Retrieve and process the selected checkboxes
        selected_options = request.form.getlist('checkbox')
        combined_list = list_combine(selected_options)
        return results(combined_list)
    return render_template('index.html')

# ...


def list_combine(selected_options):
    combined_list = []
    for option in selected_options:
        option_values = [option]  # Wrap the option in a list
        combined_list.append(option_values)
        print(combined_list)
    return combined_list


@app.route('/results')
def results(combined_list):
    finalized_products = []

    for option in combined_list:
        print(option)
        category = option[0]
        tags = option[-3:]  # Extract the last three items as tags
        if category[0] in products:
            for prod in products[category]:
                if len(set(tags).intersection(prod['tags'])) >= 2:
                    finalized_products.append(prod['name'])
        else:
            print(f"Invalid category: {category}")

    return render_template('result.html', products=finalized_products)






@app.route('/generate-clusters', methods=['GET', 'POST'])
def generate_clusters():
    if request.method == 'POST':
        # Get form data
        num_clusters = int(request.form['num_clusters'])
        algo_type = request.form['algo_type']

        print("--GOT ALGO--")
        dataset_selection = request.form['dataset']
        print("--GOT DATASET--")

        # Determine dataset selection and load dataset
        if dataset_selection == 'dataset1':
            dataset = pd.read_csv('datasets/Sales Transaction.csv')
        elif dataset_selection == 'dataset2':
            dataset = pd.read_csv('datasets/Personal Info.csv')
        elif dataset_selection == 'dataset3':
            dataset = pd.read_csv('datasets/Customer Demographics.csv')
        elif dataset_selection == 'dataset4':
            dataset = pd.read_csv('datasets/Wine Quality.csv')
        elif dataset_selection == 'dataset5':
            dataset = pd.read_csv('datasets/online retail.csv')
        elif dataset_selection == 'dataset6':
            dataset = pd.read_csv('datasets/geolocation.csv')
        elif dataset_selection == 'dataset7':
            dataset = pd.read_csv('datasets/Product Sales.csv')
        elif dataset_selection == 'dataset8':
            dataset = pd.read_csv('datasets/Sports Performance.csv')
        elif dataset_selection == 'dataset9':
            dataset = pd.read_csv('datasets/Customer-Segmentation.csv')
        elif dataset_selection == 'dataset10':
            dataset = pd.read_csv('dataset7.csv')
        else :
            return render_template('error.html')

        # Identify and remove non-numeric columns
        non_numeric_columns = []
        for column in dataset.columns:
            if not np.issubdtype(dataset[column].dtype, np.number):
                non_numeric_columns.append(column)

        # Drop rows with NaN values
        dataset = dataset.dropna()
        dataset = dataset.drop(columns=non_numeric_columns)

        # Convert remaining columns to floats
        dataset = dataset.astype(float)


        # Get feature names
        feature_names_a = dataset.columns.tolist()
        # Convert dataset to HTML table
        table_html = dataset.to_html(index=False)


        # Run clustering algorithm
        if algo_type == 'k-means':
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(dataset)
            dataset['cluster'] = kmeans.labels_
            # Get feature names
            feature_names = dataset.columns.tolist()
            # Remove 'cluster' column from feature names
            feature_names.remove('cluster')
            # Generate scatter plot
            plots = []
            # Generate scatter plot
            plots = []
            plot_dir = os.path.join('static', 'plots')
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            image_names = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    fig, ax = plt.subplots()
                    ax.scatter(dataset[feature_names[i]], dataset[feature_names[j]], c=dataset['cluster'])
                    ax.set_xlabel(feature_names[i])
                    ax.set_ylabel(feature_names[j])
                    # Save plot image to file
                    plot_file = f'plot_{feature_names[i]}_{feature_names[j]}.png'
                    plot_path = os.path.join(plot_dir, plot_file)
                    plt.savefig(plot_path, format='png')
                    plt.close(fig)
                    image_names.append(plot_file)

            # Return results HTM
            return render_template('result-cluster.html', image_names=image_names, feature_names=feature_names,feature_names_a=feature_names_a  ,table_html=table_html)

        elif algo_type == 'hierarchical':
            # Implement hierarchical clustering algorithm
            # ...
            return render_template('under_constr.html')
        elif algo_type == 'DBSCAN':
            # Implement DBSCAN clustering algorithm
            # ...
            return render_template('under_constr.html')
        else:
            return render_template('error.html')
    return render_template('error.html')


@app.route('/customerside')
def customerside():
    return render_template('form-cs.html')














app.config['STATIC_FOLDER'] = "static"
# Home Page

@app.route('/clientend')
def clientend():
    return render_template('index.html')


# Age Page
@app.route('/age')
def age():
    return render_template('age.html')

# Gender Page
@app.route('/gender')
def gender():
    return render_template('gender.html')

# Zone Page
@app.route('/zone')
def zone():
    return render_template('zone.html')

# Device Used Page
@app.route('/deviceused')
def deviceused():
    return render_template('deviceused.html')

# Network Used Page
@app.route('/networkused')
def networkused():
    return render_template('networkused.html')

# Favorite Sport Page
@app.route('/favsport')
def favsport():
    return render_template('favsport.html')






@app.route('/generate_pdf', methods=['POST','GET'])
def generate_pdf():
    # Open the image files in binary mode
    images = ['static/sample/samp1.png', 'static/sample/samp2.png','static/sample/samp3.png','static/sample/samp4.png']  # Example values

    pdf_path = 'static/pdf/output.pdf'
    pdf_canvas = canvas.Canvas(pdf_path, pagesize=A4)

    # Add the heading to the PDF
    segmentation_id = 1  # Example value
    heading_text = f'FDS PROJECT - Fully Ready Webapp'
    pdf_canvas.setFont('Helvetica-Bold', 19)
    pdf_canvas.drawCentredString(A4[0] / 2, A4[1] - 50, heading_text)

    # Add the description to the PDF
    description_text = 'This data analytics web app collects data through WiFi tunneling creating a LAN. , Developed in a timespan of 3.5 hours only '
    pdf_canvas.setFont('Helvetica', 15)
    pdf_canvas.drawCentredString(A4[0] / 2, A4[1] - 100, description_text)

    # Add the images to the PDF
    for image_path in images:
        pdf_canvas.showPage()
        image_width, image_height = 0.8 * A4[0], 0.8 * A4[1]
        image_x = (A4[0] - image_width) / 2
        image_y = (A4[1] - image_height) / 2
        pdf_canvas.drawImage(image_path, x=image_x, y=image_y, width=image_width, height=image_height,
                             preserveAspectRatio=True, anchor='c')

    # Add the profiles column to the last page of the PDF
    profiles_text = 'Team:\n\n> Anuj Dwivedi\n> Harsh Manalel\n> Divith BS\n> Ashish Patil'
    pdf_canvas.showPage()
    pdf_canvas.setFont('Helvetica', 10)
    pdf_canvas.drawCentredString(A4[0] / 2, A4[1] - 50, profiles_text)

    # Save the PDF file
    pdf_canvas.save()
    # Send the status as a response
    response_data = {'status': 'success'}
    return jsonify(response_data)

@app.route('/download_pdf')
def download_pdf():
    # Generate the PDF file
    generate_pdf()
    # Send the PDF file as a response
    file_path = os.path.join(app.config['STATIC_FOLDER'], 'pdf', 'output.pdf')
    return send_file(file_path, as_attachment=True)

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        device_used = request.form['device_used']
        network_type = request.form['network_type']
        zone = request.form['zone']
        sports = request.form['favourite_sport']


        # Read the existing data from the Excel file
        try:
            df = pd.read_excel('data.xlsx')
        except:
            df = pd.DataFrame(
                columns=['Name', 'Age', 'Gender', 'Device Used','Network Type','Zone','Sports'])

        # Append the new data to the existing data
        new_data = {'Name': name, 'Age': age, 'Gender': gender,'Device Used':device_used,'Network Type':network_type,'Zone':zone,'Sports':sports}
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        # Save the data to the Excel file
        df.to_excel('data.xlsx', index=False)

        return render_template('success.html')

    else:
        # Render the form template
        return render_template('form.html')






@app.route('/start-customer-1',methods=['GET', 'POST'])
def start_customer_1():
    # Read the data from the Excel file
    data = pd.read_excel('data.xlsx')

    # Define the folder names for the visualizations
    folders = ['Age', 'Gender', 'Device Used', 'Network Type', 'Zone', 'Sports']

    # Define the labels for the visualizations
    # Define the labels for the visualizations
    labels = {
        'Age': 'Age',
        'Gender': 'Gender',
        'Device Used': 'Device Used',
        'Network Type': 'Network Type',
        'Zone': 'Zone',
        'Sports': 'Sports'
    }

    # Create the static folders if they don't already exist
    for folder in folders:
        if not os.path.exists(f'static/{folder}'):
            os.makedirs(f'static/{folder}')

    # Generate the visualizations for each folder
    for folder in folders:
        # Get the column name for the folder
        column = folder.replace('-', '_')

        # Generate the visualizations for each data type
        for data_type in ['bar', 'pie', 'hist', 'box']:
            print("Enter type o length ")
            # Define the filename for the visualization
            filename = f'{data_type}.jpg'

            # Generate the visualization
            if data_type == 'bar':
                plt.bar(data[column].value_counts().index, data[column].value_counts())
                plt.xlabel(labels[folder])
                plt.ylabel('Count')
            elif data_type == 'pie':
                plt.pie(data[column].value_counts(), labels=data[column].value_counts().index)
            elif data_type == 'hist':
                plt.hist(data[column], bins=range(0, 51, 5))
                plt.xlabel(labels[folder])
                plt.ylabel('Frequency')
            elif data_type == 'box':
                plt.boxplot(data.groupby(column)['Age'].apply(list), labels=data[column].unique())
                plt.xlabel(labels[folder])
                plt.ylabel('Age')

            # Save the visualization to the appropriate folder
            plt.savefig(f'static/{folder}/{filename}')

            # Clear the plot for the next visualization
            plt.clf()
    return render_template('home.html')










if __name__ == '__main__':
    app.run(debug=True)
