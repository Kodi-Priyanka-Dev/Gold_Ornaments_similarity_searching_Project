# 🚀 Deployment Folder Structure

When you deploy this application to a production server (like AWS, Render, Heroku, or a local network server), you do **not** need the entire training dataset or all the training scripts (like `main.py`).

Here is the exact, minimized folder structure you need for deployment:

```text
Gold searching/
├── backend/
│   ├── api/
│   │   └── routes.py             # API endpoints
│   ├── services/
│   │   └── similarity_search.py  # ImageNet ML logic and classification check
│   ├── app.py                    # The Flask server entry point
│   ├── requirements.txt          # Python dependencies
│   └── uploads/                  # Temporary storage (created automatically)
│
├── frontend/                     
│   ├── index.html                # The web interface (Model select removed)
│   ├── style.css                 # Web styling
│   ├── script.js                 # Frontend API logic
│   └── bg.png                    # Background image for the UI
│
├── Embeddings/
│   ├── imagenet_image_features.npy  # ImageNet embeddings for similarity search
│   └── imagenet_image_paths.pkl     # Paths for the ImageNet embeddings
│
└── models/                       
    └── best_model.pth            # STILL REQUIRED! Used for the initial "Is this jewelry?" classification check
```

## Deployment Notes:
1. **Dataset:** You only need the `dataset/` folder if the frontend actually needs to display those images in the search results (which it does, since `imagenet_image_paths.pkl` points to `dataset/...`). If you want a truly minimal deployment, you would upload the dataset images to a cloud bucket (like AWS S3) and update the `.pkl` file with URL links instead of local paths. For now, keep the `dataset/` folder wherever the backend is running.
2. **Models:** Even though we switched to ImageNet for the similarity search, your backend still loads `best_model.pth` to perform the *Classification Check* (making sure the uploaded image is actually a ring/necklace/bangle and not a random object).
3. **Ignored Files:** You can safely leave behind `training_log.csv`, `main.py` scripts, and any original `.npy`/`.pkl` files in the `Embeddings/` folder that don't have `imagenet_` in their name.

---

## 🚀 Starting and Building Process

To successfully deploy and run the project on your production machine or server, follow these steps:

### 1. Environment Setup
Create a `.env` file in your project root directory (the same level as `backend/`) to configure your Qdrant Cloud connection.
```env
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 2. Install Dependencies
Make sure Python is installed on the machine, then install the required packages:
```bash
# Navigate to the backend directory where requirements.txt is located
cd backend
pip install -r requirements.txt
```

### 3. Run the Backend Server
Start the Flask server. Since the frontend is bundled statically inside Flask, running the backend handles both API requests and web UI rendering.
```bash
# From inside the backend/ directory
python app.py
```

### 4. Access the Application
- **Main Interface:** Open your browser and go to `http://localhost:5000`
- **Admin Panel:** Go to `http://localhost:5000/admin` (to upload new inventory)

*Note: If you deploy to a cloud service, ensure port `5000` is exposed or map it to port `80`.*

---

## ☁️ Deploying to Azure App Service

The easiest way to deploy this Flask application to the Azure Cloud is by using **Azure App Service** and the Azure CLI.

### Prerequisites
1. Install the [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) on your local machine.
2. Log in to your Azure account by running:
   ```bash
   az login
   ```

### 1. Prepare for Azure
Azure automatically looks for an `app.py` file to start a Flask app, which you already have. It will also automatically install dependencies from your `requirements.txt`.

Ensure your project is structured with `backend` as the root of the deployment, or deploy the entire folder and configure the startup command. The easiest way is to deploy the `backend/` folder directly.

### 2. Create and Deploy the Web App
Navigate into your `backend` directory and use the `az webapp up` command. This single command creates a resource group, an App Service plan, and deploys your code.

```bash
cd backend
az webapp up --name your-unique-app-name --runtime "PYTHON|3.10" --os-type linux
```
*(Replace `your-unique-app-name` with a globally unique name for your application).*

### 3. Configure Environment Variables
Your application requires the Qdrant API keys to function. Once the deployment finishes, you must add these to Azure's App Settings so the app can read them (this acts as your `.env` file in the cloud).

```bash
az webapp config appsettings set --name your-unique-app-name --settings QDRANT_URL="your_qdrant_cluster_url" QDRANT_API_KEY="your_qdrant_api_key"
```

### 4. Custom Startup Command (If needed)
By default, Azure will use Gunicorn to run your Flask app. If you need to specify exactly how to run it, you can set the startup command:
```bash
az webapp config set --name your-unique-app-name --startup-file "gunicorn --bind=0.0.0.0 --timeout 600 app:app"
```

### 5. Access your App
Your application will be live at:
`https://your-unique-app-name.azurewebsites.net`

---

## 🌐 Deploying Manually via Azure Portal (Web UI)

If you do not have access to the Azure CLI, you can deploy the application entirely through the Azure Web Portal.

### 1. Push your Code to GitHub
The easiest way to deploy via the portal is to connect it to a GitHub repository.
1. Push your `backend/` folder, `models/best_model.pth`, and `frontend/` folder to a new private GitHub repository.

### 2. Create the Web App in Azure
1. Go to [portal.azure.com](https://portal.azure.com/) and search for **App Services**.
2. Click **Create** > **Web App**.
3. **Basics Tab:** 
   - Choose your Subscription and Resource Group.
   - Enter a unique **Name** for your app.
   - Publish: Select **Code**.
   - Runtime stack: Select **Python 3.10**.
   - Operating System: Select **Linux**.
4. **Deployment Tab:**
   - Enable **Continuous Deployment**.
   - Connect your GitHub account and select the repository you created in Step 1.
5. Click **Review + Create** and wait for the resource to be provisioned.

### 3. Add Environment Variables (API Keys)
1. Once the App Service is created, go to the resource page.
2. In the left-hand menu, scroll down to **Settings** and click **Environment variables** (or Configuration).
3. Under **App settings**, click **New application setting**.
4. Add your variables one by one:
   - Name: `QDRANT_URL`, Value: `your_qdrant_cluster_url`
   - Name: `QDRANT_API_KEY`, Value: `your_qdrant_api_key`
5. Click **Apply** and **Save**. The app will automatically restart.

### 4. Configure Startup Command
1. In the left-hand menu, go to **Settings** > **Configuration**.
2. Click the **General settings** tab.
3. In the **Startup Command** box, enter:
   ```text
   gunicorn --bind=0.0.0.0 --timeout 600 app:app
   ```
4. Click **Save**. Azure will pull your code from GitHub, install the `requirements.txt`, and start the Flask server!
