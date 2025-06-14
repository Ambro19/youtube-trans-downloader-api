# 🚀 Complete Render Environment Variables Setup

## 📋 **Production Environment Variables for Render**

Add these to your Render service's Environment tab:

### **🔐 Authentication & Security**
```bash
SECRET_KEY=your_super_secure_production_jwt_key_minimum_32_characters_long
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
ENVIRONMENT=production
```

### **🗄️ Database**
```bash
# Render will automatically provide DATABASE_URL for PostgreSQL
# If using SQLite in production (not recommended), set:
# DATABASE_URL=sqlite:///./youtube_trans_downloader.db
```

### **💳 Stripe Configuration (LIVE KEYS)**
```bash
# Get these from Stripe Dashboard (LIVE mode, not test mode)
STRIPE_SECRET_KEY=sk_live_51xyz...your_actual_live_secret_key
STRIPE_PRO_PRICE_ID=price_xyz...your_live_pro_price_id
STRIPE_PREMIUM_PRICE_ID=price_xyz...your_live_premium_price_id
STRIPE_WEBHOOK_SECRET=whsec_xyz...your_live_webhook_secret
```

### **🌐 Domain & CORS**
```bash
DOMAIN=https://youtube-trans-downloader-api.onrender.com
FRONTEND_URL=https://your-frontend-domain.com
```

### **📧 Optional: Email Configuration**
```bash
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_app_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

## 🔍 **How to Add These in Render Dashboard**

### **Step 1: Access Environment Variables**
1. **Go to** [https://dashboard.render.com](https://dashboard.render.com)
2. **Click your service** (youtube-trans-downloader-api)
3. **Click "Environment"** in the left sidebar

### **Step 2: Add Each Variable**
For each environment variable above:
1. **Click "Add Environment Variable"**
2. **Key**: Enter the variable name (e.g., `STRIPE_SECRET_KEY`)
3. **Value**: Enter the actual value (e.g., `sk_live_51xyz...`)
4. **Click "Save Changes"**

### **Step 3: Deploy with New Variables**
After adding all variables:
1. **Click "Manual Deploy"** → **"Deploy latest commit"**
2. **Wait for deployment** to complete
3. **Check logs** - warnings should be gone!

## ⚠️ **Important Security Notes**

### **🔴 Use LIVE Keys for Production**
- ❌ **Don't use** `sk_test_` keys in production
- ✅ **Use** `sk_live_` keys for production
- ❌ **Don't use** test price IDs in production
- ✅ **Use** live price IDs for production

### **🔐 Secure Key Management**
- ✅ **Different JWT secret** for production vs development
- ✅ **Longer, more complex secrets** for production
- ✅ **Never commit** production keys to GitHub
- ✅ **Rotate keys** periodically

### **🌐 Production Webhook Setup**
Create a **separate webhook** for production:
- **URL**: `https://youtube-trans-downloader-api.onrender.com/stripe_webhook/`
- **Mode**: Live (not test)
- **Events**: `invoice.payment_succeeded`, `invoice.payment_failed`, `customer.subscription.deleted`

## 🧪 **Testing After Setup**

### **1. Check Logs**
After redeployment, logs should show:
```
✅ All required environment variables are set
🌍 Running in production mode
✅ Database initialized successfully
🎉 Application startup complete!
```

### **2. Test API Health**
Visit: `https://youtube-trans-downloader-api.onrender.com/health/`

Should return:
```json
{
  "status": "healthy", 
  "stripe_configured": true,
  "timestamp": "2025-06-08T..."
}
```

### **3. Test Subscription Flow**
1. **Frontend connects** to production API
2. **Subscription page loads** without errors
3. **Payment processing** works with real cards
4. **Webhooks** are received and processed

## 🔧 **Troubleshooting Common Issues**

### **Issue: Still seeing warnings after adding variables**
- **Solution**: Make sure you clicked "Save Changes" and redeployed

### **Issue: Stripe errors in production**
- **Check**: You're using LIVE keys, not test keys
- **Check**: Price IDs are from LIVE mode in Stripe
- **Check**: Webhook URL points to your production domain

### **Issue: Database errors**
- **Check**: DATABASE_URL is properly set
- **Consider**: Upgrading to PostgreSQL for production

### **Issue: CORS errors**
- **Check**: FRONTEND_URL points to your actual frontend domain
- **Update**: CORS allowed origins in your code

## 📊 **Expected Results**

After completing this setup:
- ✅ **No more warning messages** in logs
- ✅ **Subscription payments work** with real credit cards
- ✅ **Webhooks are processed** correctly
- ✅ **Production-ready** Stripe integration
- ✅ **Secure environment** configuration

## 🎯 **Quick Checklist**

- [ ] Added all required environment variables to Render
- [ ] Used LIVE Stripe keys (not test keys)
- [ ] Created production webhook endpoint
- [ ] Redeployed service after adding variables
- [ ] Tested API health endpoint
- [ ] Verified no warnings in logs
- [ ] Tested subscription flow end-to-end