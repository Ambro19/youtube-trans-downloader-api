# 🚀 Complete Stripe Subscription Implementation Guide

This guide will help you implement the full Pro/Premium subscription system for your YouTube Transcript Downloader.

## 📋 Prerequisites

- Stripe account (create at https://stripe.com)
- Node.js and Python environment set up
- Your existing YouTube Transcript Downloader project

## 🔧 Step 1: Stripe Configuration

### 1.1 Set Up Stripe Account

1. Create a Stripe account and verify your email
2. Go to **Developers > API keys**
3. Copy your **Publishable key** and **Secret key** (use test keys for development)

### 1.2 Create Products and Prices

1. Go to **Products** in your Stripe Dashboard
2. Create two products:

**Pro Plan:**
- Name: "Pro Plan"  
- Description: "100 downloads per month with priority processing"
- Price: $9.99/month (recurring)
- Copy the **Price ID** (starts with `price_`)

**Premium Plan:**
- Name: "Premium Plan"
- Description: "Unlimited downloads with fastest processing"  
- Price: $19.99/month (recurring)
- Copy the **Price ID** (starts with `price_`)

### 1.3 Set Up Webhooks

1. Go to **Developers > Webhooks**
2. Add endpoint: `http://localhost:8000/stripe_webhook/`
3. Select these events:
   - `invoice.payment_succeeded`
   - `invoice.payment_failed` 
   - `customer.subscription.deleted`
4. Copy the **Webhook Secret** (starts with `whsec_`)

## 🗄️ Step 2: Database Migration

### 2.1 Update Database Schema

Run the SQL migration to add subscription fields:

```sql
-- Add to your database (SQLite/PostgreSQL/MySQL)
ALTER TABLE users ADD COLUMN subscription_tier VARCHAR(20) DEFAULT 'free';
ALTER TABLE users ADD COLUMN subscription_status VARCHAR(20) DEFAULT 'inactive';
ALTER TABLE users ADD COLUMN subscription_id VARCHAR(255);
ALTER TABLE users ADD COLUMN subscription_current_period_end DATETIME;
ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR(255);

-- Usage tracking
ALTER TABLE users ADD COLUMN usage_clean_transcripts INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN usage_unclean_transcripts INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN usage_audio_downloads INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN usage_video_downloads INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN usage_reset_date DATETIME DEFAULT CURRENT_TIMESTAMP;
```

### 2.2 Update User Model

Replace your User model with the enhanced version that includes subscription fields.

## 🔑 Step 3: Environment Variables

### 3.1 Frontend (.env)

Create `.env` in your React app root:

```bash
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key
REACT_APP_PRO_PRICE_ID=price_your_pro_price_id
REACT_APP_PREMIUM_PRICE_ID=price_your_premium_price_id
```

### 3.2 Backend (.env)

Create `.env` in your Python backend:

```bash
DATABASE_URL=sqlite:///./youtube_trans_downloader.db
SECRET_KEY=your_super_secret_jwt_key
STRIPE_SECRET_KEY=sk_test_your_secret_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret
STRIPE_PRO_PRICE_ID=price_your_pro_price_id
STRIPE_PREMIUM_PRICE_ID=price_your_premium_price_id
```

## 📦 Step 4: Install Dependencies

### 4.1 Frontend Dependencies

```bash
npm install @stripe/stripe-js @stripe/react-stripe-js react-hot-toast
```

### 4.2 Backend Dependencies

```bash
pip install stripe python-multipart
```

## 🔄 Step 5: Implement Files

### 5.1 Frontend Files

1. **Replace** `src/components/PaymentForm.js` with the enhanced version
2. **Replace** `src/contexts/SubscriptionContext.js` with the enhanced version  
3. **Add** `src/config/config.js` for configuration
4. **Update** your existing `SubscriptionPage.js` to use the new context

### 5.2 Backend Files

1. **Add** `payment.py` with all payment processing functions
2. **Update** `main.py` to include the new API routes
3. **Replace** your User model with the enhanced version
4. **Run** the database migration

## 🧪 Step 6: Testing

### 6.1 Test Cards (Stripe Test Mode)

Use these test card numbers:

- **Success:** 4242 4242 4242 4242
- **Decline:** 4000 0000 0000 0002  
- **Any future expiry date (e.g., 12/30)**
- **Any 3-digit CVC (e.g., 123)**

### 6.2 Test Flow

1. Register a new user account
2. Go to subscription page
3. Click "Upgrade to Pro"
4. Enter test card details
5. Verify subscription is created
6. Test download limits
7. Test cancellation

## 🔍 Step 7: Verification

### 7.1 Check Stripe Dashboard

- Verify customers are created
- Check subscriptions are active
- Monitor webhook events

### 7.2 Test Frontend

1. **Subscription Status:** Check current plan displays correctly
2. **Usage Tracking:** Verify counters update after downloads  
3. **Limit Enforcement:** Test that limits are enforced
4. **Payment Flow:** Ensure payment modal works
5. **Cancellation:** Test subscription cancellation

## 🚨 Common Issues & Fixes

### Issue 1: "Subscribe Now" Button Not Working

**Solution:** 
- Check Stripe publishable key is correct
- Verify PaymentForm component is imported correctly
- Check browser console for JavaScript errors

### Issue 2: Payment Intent Creation Fails

**Solution:**
- Verify Stripe secret key is set in backend
- Check that payment.py is imported in main.py
- Ensure CORS is configured for your frontend domain

### Issue 3: Subscription Not Created

**Solution:**
- Check webhook endpoint is accessible
- Verify price IDs match between frontend and backend
- Test webhook delivery in Stripe dashboard

### Issue 4: Usage Limits Not Working

**Solution:**
- Ensure database migration was applied
- Check that usage increment functions are called
- Verify subscription context is properly providing data

## 🔄 Step 8: Production Deployment

### 8.1 Switch to Live Mode

1. Replace test keys with live keys
2. Update webhook endpoint to production URL
3. Test with real card (small amount)
4. Set up SSL certificate for webhook security

### 8.2 Security Checklist

- [ ] Environment variables are secure
- [ ] Webhook endpoints verify Stripe signatures  
- [ ] Database has proper indexes
- [ ] Error logging is configured
- [ ] Rate limiting is in place

## 📊 Step 9: Monitoring

### 9.1 Key Metrics to Track

- Subscription conversion rate
- Monthly recurring revenue (MRR)
- Churn rate
- Usage patterns by tier

### 9.2 Set Up Alerts

- Failed payments
- Webhook delivery failures
- High usage approaching limits
- Subscription cancellations

## 🎯 Next Steps

After implementing the basic subscription system, consider:

1. **Analytics Dashboard:** Track subscription metrics
2. **Email Notifications:** Payment confirmations, usage alerts
3. **Annual Plans:** Offer discounted yearly subscriptions  
4. **Usage Analytics:** Help users optimize their usage
5. **API Access:** Implement API keys for Premium users

## 🆘 Need Help?

If you encounter issues:

1. Check the browser console for errors
2. Verify environment variables are set correctly
3. Test webhook delivery in Stripe dashboard
4. Check your backend logs for detailed error messages

---

**🎉 Congratulations!** You now have a complete subscription system with:
- ✅ Working Stripe payment integration
- ✅ Pro and Premium plan upgrades  
- ✅ Usage tracking and limit enforcement
- ✅ Subscription management (cancel/upgrade)
- ✅ Real-time usage updates

Your users can now upgrade from Free → Pro → Premium with full payment processing!