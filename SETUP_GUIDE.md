# YouTube Transcript Downloader Setup Guide

## Project Structure
```
YOUTUBE-TRANSCRIPT-APP/
├── backend/
│   ├── database.py
│   ├── main.py
│   ├── .env
│   ├── requirements.txt
│   └── run.py
└── frontend/
    ├── package.json
    ├── .env                    # CREATE THIS FILE
    ├── .env.example
    ├── src/
    │   ├── config/
    │   │   └── config.js
    │   ├── contexts/
    │   ├── pages/
    │   ├── components/
    │   └── ...
    └── ...
```

## Frontend Setup Instructions

### 1. Navigate to Frontend Directory
```bash
cd frontend
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Configure Environment Variables

#### Step 3a: Create the .env file
Copy the example environment file:
```bash
cp .env.example .env
```

#### Step 3b: Update .env with your actual values
Open `frontend/.env` in your text editor and update:

```bash
# API Configuration - Update if your backend runs on different port
REACT_APP_API_URL=http://localhost:8000

# Stripe Configuration - Replace with YOUR actual Stripe publishable key
REACT_APP_STRIPE_PUBLISHABLE_KEY=pk_test_your_actual_stripe_key_goes_here

# App Configuration
REACT_APP_NAME=YouTube Transcript Downloader
```

### 4. Get Your Stripe Publishable Key

1. Log into your Stripe Dashboard (https://dashboard.stripe.com)
2. Go to "Developers" → "API keys"
3. Copy your **Publishable key** (starts with `pk_test_` for test mode)
4. Replace `pk_test_your_actual_stripe_key_goes_here` in your `.env` file

**Example:**
```bash
REACT_APP_STRIPE_PUBLISHABLE_KEY=pk_test_51234567890abcdefghijklmnop...
```

### 5. Start the Development Server
```bash
npm start
```

The app will open at http://localhost:3000

## Backend Setup Instructions

### 1. Navigate to Backend Directory
```bash
cd backend
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Backend Environment
Update `backend/.env`:
```bash
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key_here
SECRET_KEY=your-secure-jwt-secret-key-here
```

### 4. Start the Backend Server
```bash
python run.py
```

The API will be available at http://localhost:8000

## Environment Variables Explained

### Frontend (.env)
- `REACT_APP_API_URL`: The URL where your FastAPI backend is running
- `REACT_APP_STRIPE_PUBLISHABLE_KEY`: Your Stripe publishable key (safe to expose in frontend)
- `REACT_APP_NAME`: Application name displayed in the UI

### Backend (.env)
- `STRIPE_SECRET_KEY`: Your Stripe secret key (keep this private!)
- `SECRET_KEY`: JWT token signing key (generate a secure random string)

## Troubleshooting

### Frontend Issues
1. **Stripe not loading**: Check that your publishable key is correct and starts with `pk_test_`
2. **API connection failed**: Ensure backend is running on the specified API_URL
3. **Environment variables not working**: Make sure they start with `REACT_APP_`

### Backend Issues
1. **Stripe errors**: Verify your secret key is correct and starts with `sk_test_`
2. **Database errors**: The SQLite database will be created automatically on first run

### Common Setup Mistakes
1. Using the **secret key** in frontend (should be **publishable key**)
2. Forgetting the `REACT_APP_` prefix for frontend environment variables
3. Not restarting the server after changing environment variables

## Testing the Setup

1. Register a new account at http://localhost:3000/register
2. Login with your credentials
3. Try downloading a transcript (you'll have 5 free unclean transcripts)
4. Test the subscription flow with Stripe test cards:
   - Success: `4242 4242 4242 4242`
   - Failure: `4000 0000 0000 0002`

## Production Deployment Notes

For production:
1. Use live Stripe keys (starting with `pk_live_` and `sk_live_`)
2. Update `REACT_APP_API_URL` to your production backend URL
3. Use a secure random string for `SECRET_KEY`
4. Consider using environment-specific configuration