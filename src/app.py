from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# --- Load Model Assets ---
# This part runs only once when the server starts
try:
    cosine_sim_matrix = np.load('model_assets/cosine_sim_matrix.npy')
    products = pd.read_csv('model_assets/products.csv')

    # Create a mapping of product names to their DataFrame index for quick lookup
    indices = pd.Series(products.index, index=products['product_name']).drop_duplicates()
    
except FileNotFoundError:
    print("Error: Model assets not found. Please run the model storing step first.")
    cosine_sim_matrix = None
    products = None
    indices = None

# --- Recommendation Function (re-used from your project) ---
def get_recommendations_for_flask(product_name, cosine_sim, product_df, indices, num_recommendations=10):
    if product_name not in indices:
        return []

    idx = indices[product_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    
    product_indices = [i[0] for i in sim_scores]
    return product_df.iloc[product_indices]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    print(request.form)  # Debugging line to check form data
    # Get the product name from the form
    product_name = request.form.get('product_name')

    # Check if the product name is empty or if the model isn't loaded
    if not product_name or not products:
        # Redirect back to the home page or render an error message
        return render_template('index.html', error="Please enter a product name.")
        
    # --- The rest of your recommendation logic here ---
    recommendations = get_recommendations_for_flask(product_name, cosine_sim_matrix, products, indices, num_recommendations=5)
    
    if not recommendations.empty:
        return render_template('index.html', recommendations=recommendations.to_dict('records'), input_product=product_name)
    else:
        return render_template('index.html', error="Product not found. Please try another name.", input_product=product_name)
if __name__ == '__main__':
    app.run(debug=True)