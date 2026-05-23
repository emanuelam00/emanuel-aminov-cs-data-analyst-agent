"""
Download the Bitext Customer Service dataset from HuggingFace and save it locally.
Run once before starting the agent: python download_data.py

If HuggingFace is unavailable, the script falls back to generating a small
synthetic dataset with the same schema so the agent still works for demos.
"""

import os
import random
import pandas as pd
from datasets import load_dataset


DATASET_PATH = os.path.join("data", "bitext_dataset.csv")


# ── Fallback synthetic data ─────────────────────────────────────────────────

SYNTHETIC_DATA = {
    "ACCOUNT": {
        "create_account": [
            ("I need to create a new account.", "I'd be happy to help you create a new account. Please visit our registration page and fill in your details."),
            ("How do I sign up?", "You can sign up by clicking the 'Register' button on our homepage and completing the form."),
            ("I want to open an account with you.", "Welcome! To open an account, please provide your name, email address, and a secure password on our sign-up page."),
        ],
        "edit_account": [
            ("I need to update my email address.", "To update your email, go to Account Settings and click on 'Edit Profile'."),
            ("How can I change my password?", "You can change your password in Account Settings under the Security tab."),
            ("I'd like to update my billing information.", "To update billing info, navigate to Account Settings and select 'Payment Methods'."),
        ],
        "delete_account": [
            ("I want to delete my account.", "I'm sorry to hear that. To delete your account, go to Settings > Privacy > Delete Account."),
            ("How do I close my account permanently?", "Account closure is permanent. Please go to Settings > Account > Close Account and follow the prompts."),
        ],
        "recover_password": [
            ("I forgot my password.", "Click 'Forgot Password' on the login page and we'll send a reset link to your email."),
            ("I can't log in, I lost my password.", "No worries! Use the 'Reset Password' option on the login page and follow the instructions sent to your email."),
        ],
    },
    "REFUND": {
        "get_refund": [
            ("I want a refund for my order.", "I understand. Please provide your order number and I'll process the refund right away."),
            ("I'd like my money back for this purchase.", "I'm sorry for the inconvenience. Refunds are processed within 5-7 business days once approved."),
            ("This product didn't work, I need a refund.", "I apologise for the trouble. I'll initiate a refund for you immediately."),
        ],
        "track_refund": [
            ("Where is my refund?", "Your refund is being processed and should appear in your account within 3-5 business days."),
            ("I haven't received my refund yet.", "I can check the status of your refund. Could you provide your order number?"),
        ],
        "refund_policy": [
            ("What is your refund policy?", "We offer a 30-day money-back guarantee on all purchases. Items must be unused and in original condition."),
            ("How long does a refund take?", "Refunds are processed within 5-7 business days after approval."),
        ],
    },
    "SHIPPING": {
        "track_order": [
            ("Where is my package?", "Let me track your package. Could you provide your order number?"),
            ("I want to track my shipment.", "You can track your shipment using the tracking number in your confirmation email."),
            ("My order hasn't arrived yet.", "I apologise for the delay. Let me look into the status of your shipment."),
        ],
        "delivery_options": [
            ("What shipping options do you offer?", "We offer standard (5-7 days), express (2-3 days), and overnight shipping."),
            ("How fast can you deliver?", "Our fastest option is overnight shipping, available for most locations."),
        ],
        "shipping_address": [
            ("I need to change my delivery address.", "To change your delivery address, please contact us as soon as possible before dispatch."),
            ("Can I update my shipping address?", "If your order hasn't shipped yet, we can update the address. Please provide your order number."),
        ],
    },
    "ORDER": {
        "place_order": [
            ("I'd like to place an order.", "I'd be happy to help you place an order. What product are you interested in?"),
            ("How do I order something?", "You can place an order by adding items to your cart and proceeding to checkout."),
        ],
        "cancel_order": [
            ("I want to cancel my order.", "I can help you cancel your order. Please provide your order number."),
            ("Please cancel my purchase.", "I'll process the cancellation immediately. Your refund will be issued within 5-7 business days."),
        ],
        "order_status": [
            ("What's the status of my order?", "Your order is currently being processed and will ship within 1-2 business days."),
            ("Is my order confirmed?", "Yes, your order is confirmed. You'll receive a shipping notification once it dispatches."),
        ],
    },
    "FEEDBACK": {
        "complaint": [
            ("I'm very unhappy with the service I received.", "I sincerely apologise for your experience. I'll escalate this to our quality team immediately."),
            ("This is unacceptable, I'm making a complaint.", "I'm truly sorry to hear this. Your feedback is important and I'll make sure it reaches the right team."),
            ("Your service has been terrible.", "I apologise for the inconvenience you've experienced. We take all complaints seriously."),
        ],
        "review": [
            ("I'd like to leave a review.", "Thank you for wanting to share your feedback! You can leave a review on our website's product page."),
            ("Where can I rate your service?", "You can rate our service on our website or through the feedback link in your order confirmation email."),
        ],
    },
    "PAYMENT": {
        "payment_issue": [
            ("My payment didn't go through.", "I'm sorry to hear that. Could you check if your card details are correct and try again?"),
            ("I'm having trouble paying.", "Let me help you with the payment issue. Could you describe what error you're seeing?"),
        ],
        "payment_methods": [
            ("What payment methods do you accept?", "We accept Visa, MasterCard, American Express, PayPal, and bank transfers."),
            ("Can I pay with PayPal?", "Yes, we accept PayPal as a payment method at checkout."),
        ],
        "invoice": [
            ("I need an invoice for my order.", "I can generate an invoice for you. Please provide your order number."),
            ("Where can I find my receipt?", "Your invoice is available in your account under 'Order History'."),
        ],
    },
    "CANCELLATION_FEE": {
        "check_cancellation_fee": [
            ("Is there a cancellation fee?", "Cancellation fees depend on the timing. Cancellations within 24 hours are free."),
            ("Will I be charged for cancelling?", "For cancellations after 24 hours of purchase, a 10% fee may apply."),
        ],
        "waive_cancellation_fee": [
            ("Can you waive the cancellation fee?", "I understand your concern. Let me review your case to see if an exception can be made."),
            ("I shouldn't have to pay a cancellation fee.", "I'm sorry for the inconvenience. I'll look into waiving the fee given the circumstances."),
        ],
    },
    "CONTACT": {
        "contact_customer_service": [
            ("I need to speak to someone.", "You can reach our support team by phone, email, or live chat 24/7."),
            ("How can I contact you?", "You can contact us via email at support@company.com or call our helpline."),
        ],
        "contact_human_agent": [
            ("I want to talk to a real person.", "I understand. I'll transfer you to a human agent right away."),
            ("Can I speak to a manager?", "Of course. I'll connect you with a supervisor immediately."),
        ],
    },
    "NEWSLETTER": {
        "newsletter_subscription": [
            ("I want to subscribe to your newsletter.", "Great! You can subscribe by entering your email on our website's newsletter section."),
            ("Sign me up for emails.", "I've added you to our mailing list. You'll receive our latest updates and offers."),
        ],
        "newsletter_unsubscription": [
            ("Please unsubscribe me from your newsletter.", "I've processed your unsubscribe request. You'll no longer receive our emails."),
            ("Stop sending me emails.", "I apologise for any inconvenience. I've removed you from our mailing list immediately."),
        ],
    },
    "INVOICE": {
        "get_invoice": [
            ("I need a copy of my invoice.", "I'll send you a copy of your invoice to your registered email address right away."),
            ("Can you send me my receipt?", "Of course! Your invoice will be emailed to you within a few minutes."),
        ],
        "wrong_invoice": [
            ("There's an error on my invoice.", "I apologise for the error. Could you describe what's incorrect so I can fix it?"),
            ("My invoice has the wrong amount.", "I'm sorry about that. I'll look into the discrepancy and issue a corrected invoice."),
        ],
    },
    "DELIVERY": {
        "delivery_period": [
            ("When will my order arrive?", "Standard delivery takes 5-7 business days. Express delivery takes 2-3 business days."),
            ("How long does delivery take?", "Delivery times vary by location. Typically 3-7 business days for domestic orders."),
        ],
        "lost_or_missing_order": [
            ("My order is lost.", "I'm very sorry to hear that. I'll file a lost shipment claim and send a replacement immediately."),
            ("I never received my package.", "I apologise for this. Let me investigate with the courier and arrange a reshipment."),
        ],
    },
}


def generate_synthetic_dataset(output_path: str) -> pd.DataFrame:
    """Generate a synthetic Bitext-like dataset when HuggingFace is unavailable."""
    rows = []
    for category, intents in SYNTHETIC_DATA.items():
        for intent, pairs in intents.items():
            for instruction, response in pairs:
                rows.append({
                    "flags": "B",
                    "instruction": instruction,
                    "category": category,
                    "intent": intent,
                    "response": response,
                })
    # Shuffle for realism
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def download_dataset() -> None:
    """Download and save the Bitext customer service dataset to data/bitext_dataset.csv.

    Falls back to a synthetic dataset if HuggingFace is unreachable.
    """
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at {DATASET_PATH}. Skipping download.")
        df = pd.read_csv(DATASET_PATH)
        print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")
        return

    print("Downloading Bitext Customer Service dataset from HuggingFace...")
    try:
        dataset = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        )
        df = dataset["train"].to_pandas()
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f"  Saved to {DATASET_PATH}")
        print(f"  Rows: {len(df)} | Categories: {sorted(df['category'].unique().tolist())}")
    except Exception as exc:
        print(f"  HuggingFace download failed ({exc}). Generating synthetic dataset...")
        df = generate_synthetic_dataset(DATASET_PATH)
        print(f"  Synthetic dataset saved to {DATASET_PATH}")
        print(f"  Rows: {len(df)} | Categories: {sorted(df['category'].unique().tolist())}")


if __name__ == "__main__":
    download_dataset()
