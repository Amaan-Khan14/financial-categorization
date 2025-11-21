"""
Sample Transaction Data Generator with Dates and Amounts

Generates realistic transaction history for insights demo
"""
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """Generate realistic transaction data for insights demo"""

    # Indian merchant names by category
    MERCHANTS = {
        'Food & Dining': [
            'Swiggy food delivery', 'Zomato order', 'Dominos pizza',
            'McDonald\'s', 'KFC chicken', 'Cafe Coffee Day',
            'Starbucks coffee', 'Haldirams restaurant', 'Barbeque Nation',
            'Pizza Hut', 'Subway sandwich', 'Faasos wrap'
        ],
        'Groceries': [
            'BigBasket grocery', 'DMart shopping', 'JioMart delivery',
            'Blinkit instant delivery', 'Zepto groceries', 'Dunzo essentials',
            'More Megastore', 'Reliance Fresh', 'Spencer\'s Retail'
        ],
        'Fuel': [
            'Indian Oil petrol', 'BPCL fuel', 'HPCL pump',
            'Reliance Petroleum', 'Shell petrol'
        ],
        'Shopping': [
            'Myntra fashion', 'Flipkart purchase', 'Amazon India',
            'Nykaa beauty', 'AJIO clothing', 'Meesho shopping',
            'Lifestyle store', 'Reliance Digital', 'Croma electronics'
        ],
        'Utilities': [
            'Airtel mobile recharge', 'Jio bill payment', 'Vodafone recharge',
            'Electricity bill', 'Water bill', 'Gas cylinder',
            'Broadband bill', 'DTH recharge'
        ],
        'Entertainment': [
            'Netflix subscription', 'Amazon Prime Video', 'Disney+ Hotstar',
            'Spotify Premium', 'BookMyShow tickets', 'PVR Cinemas',
            'Gaana Plus', 'YouTube Premium', 'Sony LIV'
        ],
        'Transportation': [
            'Ola cab ride', 'Uber trip', 'IRCTC train ticket',
            'RedBus ticket', 'Metro card recharge', 'Rapido bike',
            'Auto rickshaw', 'Bus ticket'
        ],
        'Health': [
            'Apollo Pharmacy', 'Netmeds order', 'PharmEasy medicines',
            '1mg medicine', 'Practo consultation', 'Lab test',
            'Fortis Hospital', 'Max Healthcare'
        ]
    }

    # Typical amount ranges by category (in INR)
    AMOUNT_RANGES = {
        'Food & Dining': (150, 1200),
        'Groceries': (500, 3500),
        'Fuel': (500, 2000),
        'Shopping': (300, 5000),
        'Utilities': (200, 1500),
        'Entertainment': (99, 799),
        'Transportation': (50, 800),
        'Health': (200, 3000)
    }

    # Category frequency weights (how often each category appears)
    CATEGORY_WEIGHTS = {
        'Food & Dining': 0.25,      # 25% - Most frequent
        'Groceries': 0.15,           # 15%
        'Shopping': 0.15,            # 15%
        'Transportation': 0.15,      # 15%
        'Utilities': 0.10,           # 10%
        'Entertainment': 0.08,       # 8%
        'Fuel': 0.07,                # 7%
        'Health': 0.05               # 5% - Least frequent
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed"""
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        logger.info(f"Initialized SampleDataGenerator with seed={seed}")

    def generate_transaction_history(
        self,
        num_transactions: int = 100,
        days_back: int = 90
    ) -> List[Dict]:
        """
        Generate realistic transaction history

        Args:
            num_transactions: Number of transactions to generate
            days_back: How many days of history to generate

        Returns:
            List of transaction dicts with transaction_id, transaction, category, amount, date
        """
        transactions = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Generating {num_transactions} transactions from {start_date.date()} to {end_date.date()}")

        categories = list(self.MERCHANTS.keys())
        weights = [self.CATEGORY_WEIGHTS[cat] for cat in categories]

        for i in range(num_transactions):
            # Select category based on weights
            category = random.choices(categories, weights=weights)[0]

            # Select random merchant from category
            merchant = random.choice(self.MERCHANTS[category])

            # Generate random date
            random_days = random.randint(0, days_back)
            transaction_date = start_date + timedelta(days=random_days)

            # Generate amount with some variation
            min_amt, max_amt = self.AMOUNT_RANGES[category]
            # Use log-normal distribution for more realistic amounts
            mean_amt = (min_amt + max_amt) / 2
            std_amt = (max_amt - min_amt) / 4
            amount = np.random.lognormal(np.log(mean_amt), 0.3)
            amount = np.clip(amount, min_amt, max_amt)
            amount = round(amount, 2)

            # Weekend spending boost (20% higher on average)
            if transaction_date.weekday() in [5, 6]:  # Saturday, Sunday
                if category in ['Food & Dining', 'Entertainment', 'Shopping']:
                    amount *= random.uniform(1.1, 1.3)
                    amount = round(amount, 2)

            # Create transaction
            transaction = {
                'transaction_id': f"txn_{i+1:04d}",
                'transaction': merchant,
                'category': category,
                'amount': float(amount),
                'date': transaction_date.strftime('%Y-%m-%d'),
                'confidence': round(random.uniform(0.85, 0.99), 4)
            }

            transactions.append(transaction)

        # Sort by date
        transactions = sorted(transactions, key=lambda x: x['date'])

        # Add some intentional anomalies for demo (5% of transactions)
        num_anomalies = max(1, int(num_transactions * 0.05))
        anomaly_indices = random.sample(range(len(transactions)), num_anomalies)

        for idx in anomaly_indices:
            tx = transactions[idx]
            anomaly_type = random.choice(['high_amount', 'duplicate', 'unusual_day'])

            if anomaly_type == 'high_amount':
                # Make amount 3-5x higher
                tx['amount'] = round(tx['amount'] * random.uniform(3, 5), 2)
            elif anomaly_type == 'duplicate':
                # Add duplicate transaction on same day
                duplicate = tx.copy()
                duplicate['transaction_id'] = f"txn_dup_{idx}"
                transactions.insert(idx + 1, duplicate)

        logger.info(f"Generated {len(transactions)} transactions with {num_anomalies} intentional anomalies")

        return transactions

    def get_summary_stats(self, transactions: List[Dict]) -> Dict:
        """Get summary statistics of generated data"""
        if not transactions:
            return {}

        total_amount = sum(tx['amount'] for tx in transactions)
        categories = set(tx['category'] for tx in transactions)
        dates = [tx['date'] for tx in transactions]

        return {
            'total_transactions': len(transactions),
            'total_amount': round(total_amount, 2),
            'average_amount': round(total_amount / len(transactions), 2),
            'categories': sorted(list(categories)),
            'date_range': {
                'start': min(dates),
                'end': max(dates)
            }
        }
