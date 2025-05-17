import stripe
from fastapi import BackgroundTasks

stripe.api_key = "your-stripe-secret-key"  # Store in environment variables

class PaymentRequest(BaseModel):
    token: str
    subscription_tier: str
    user_id: int

@app.post("/create_subscription/")
async def create_subscription(
    request: PaymentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        # Create Stripe customer & subscription
        customer = stripe.Customer.create(
            source=request.token,
            email=user.email  # You would get this from your DB
        )
        
        # Map your tiers to Stripe price IDs
        price_id_map = {
            "basic": "price_basic_id_from_stripe",
            "premium": "price_premium_id_from_stripe"
        }
        
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{"price": price_id_map[request.subscription_tier]}]
        )
        
        # Save subscription details to database
        new_subscription = Subscription(
            user_id=request.user_id,
            tier=request.subscription_tier,
            start_date=datetime.now(),
            expiry_date=datetime.now() + timedelta(days=30),
            payment_id=subscription.id,
            auto_renew=True
        )
        db.add(new_subscription)
        db.commit()
        
        return {"status": "success", "subscription_id": subscription.id}
    
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))