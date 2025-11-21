"""
AI-Powered Financial Insights Engine

Provides:
- Anomaly detection
- Spending pattern analysis
- Forecasting
- Personalized recommendations
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class InsightsEngine:
    """
    Core insights engine for financial transaction analysis
    """

    def __init__(self, transactions: List[Dict]):
        """
        Initialize with transaction data

        Args:
            transactions: List of transaction dicts with keys:
                - transaction_id, transaction, category, amount, date, (optional) confidence
        """
        self.df = pd.DataFrame(transactions)

        if len(self.df) == 0:
            raise ValueError("No transactions provided for analysis")

        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Extract time-based features
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['day_of_week_num'] = self.df['date'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week_num'].isin([5, 6])
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['hour'] = 12  # Default to noon if no time specified

        logger.info(f"Initialized InsightsEngine with {len(self.df)} transactions")

    def detect_anomalies(self, contamination: float = 0.1) -> List[Dict]:
        """
        Detect anomalous transactions using Isolation Forest

        Args:
            contamination: Expected proportion of anomalies (0.1 = 10%)

        Returns:
            List of anomalous transactions with reasons
        """
        try:
            # Prepare features for anomaly detection
            # Normalize amounts by category to detect relative anomalies
            df_with_zscore = self.df.copy()
            df_with_zscore['amount_zscore'] = df_with_zscore.groupby('category')['amount'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

            # Features: amount z-score, day of week, is_weekend
            features = df_with_zscore[['amount_zscore', 'day_of_week_num']].values

            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )

            predictions = iso_forest.fit_predict(features)
            scores = iso_forest.score_samples(features)

            # Add predictions to dataframe
            df_with_zscore['is_anomaly'] = predictions
            df_with_zscore['anomaly_score'] = scores

            # Filter anomalies
            anomalies = df_with_zscore[df_with_zscore['is_anomaly'] == -1].copy()

            # Generate reasons for each anomaly
            results = []
            for _, row in anomalies.iterrows():
                reasons = []
                severity = "MEDIUM"

                # Check amount anomaly
                if abs(row['amount_zscore']) > 2:
                    category_mean = self.df[self.df['category'] == row['category']]['amount'].mean()
                    if row['amount'] > category_mean * 2:
                        reasons.append(f"Amount is {row['amount']/category_mean:.1f}x higher than average for {row['category']}")
                        severity = "HIGH"
                    elif row['amount'] > category_mean * 1.5:
                        reasons.append(f"Amount is {row['amount']/category_mean:.1f}x higher than usual")
                        severity = "MEDIUM"
                    elif row['amount'] < category_mean * 0.5:
                        reasons.append(f"Unusually low amount for {row['category']}")
                        severity = "LOW"

                # Check weekend vs weekday pattern
                if row['is_weekend']:
                    weekend_avg = self.df[self.df['is_weekend']]['amount'].mean()
                    if row['amount'] > weekend_avg * 1.5:
                        reasons.append("High spending on weekend")

                # Check for duplicate/similar transactions on same day
                same_day = self.df[
                    (self.df['date'] == row['date']) &
                    (self.df['category'] == row['category']) &
                    (self.df.index != row.name)
                ]
                if len(same_day) > 0:
                    reasons.append(f"Multiple {row['category']} transactions on same day")
                    severity = "HIGH"

                reason = "; ".join(reasons) if reasons else "Statistical anomaly detected"

                results.append({
                    'transaction_id': row['transaction_id'],
                    'transaction': row['transaction'],
                    'category': row['category'],
                    'amount': float(row['amount']),
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'anomaly_score': float(row['anomaly_score']),
                    'reason': reason,
                    'severity': severity
                })

            logger.info(f"Detected {len(results)} anomalies out of {len(self.df)} transactions")
            return results

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []

    def analyze_spending_patterns(self) -> Dict:
        """
        Analyze spending patterns by category and time

        Returns:
            Dict with spending_by_category and spending_by_day
        """
        try:
            # Spending by category
            category_analysis = []
            total_spending = self.df['amount'].sum()

            for category in self.df['category'].unique():
                cat_df = self.df[self.df['category'] == category]
                total = cat_df['amount'].sum()
                count = len(cat_df)
                avg = cat_df['amount'].mean()
                pct = (total / total_spending) * 100

                category_analysis.append({
                    'category': category,
                    'total_amount': float(total),
                    'transaction_count': int(count),
                    'average_amount': float(avg),
                    'percentage_of_total': float(pct)
                })

            # Sort by total amount descending
            category_analysis = sorted(category_analysis, key=lambda x: x['total_amount'], reverse=True)

            # Spending by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_analysis = []

            for day in day_order:
                day_df = self.df[self.df['day_of_week'] == day]
                if len(day_df) > 0:
                    avg = day_df['amount'].mean()
                    count = len(day_df)
                    day_analysis.append({
                        'day': day,
                        'average_spending': float(avg),
                        'transaction_count': int(count)
                    })

            # Weekend vs Weekday comparison
            weekend_spending = self.df[self.df['is_weekend']]['amount'].mean()
            weekday_spending = self.df[~self.df['is_weekend']]['amount'].mean()

            return {
                'spending_by_category': category_analysis,
                'spending_by_day': day_analysis,
                'weekend_vs_weekday': {
                    'weekend_avg': float(weekend_spending),
                    'weekday_avg': float(weekday_spending),
                    'difference_pct': float(((weekend_spending - weekday_spending) / weekday_spending) * 100)
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing spending patterns: {e}")
            return {
                'spending_by_category': [],
                'spending_by_day': [],
                'weekend_vs_weekday': {}
            }

    def forecast_spending(self) -> List[Dict]:
        """
        Forecast next month's spending by category using moving average

        Returns:
            List of forecasts per category
        """
        try:
            forecasts = []

            # Get current month spending
            latest_date = self.df['date'].max()
            current_month_start = latest_date.replace(day=1)
            current_month_df = self.df[self.df['date'] >= current_month_start]

            # Calculate days in current month so far
            days_in_current_month = (latest_date - current_month_start).days + 1
            days_in_full_month = 30  # Approximate

            for category in self.df['category'].unique():
                cat_df = self.df[self.df['category'] == category]

                # Current month spending
                current_month_cat = current_month_df[current_month_df['category'] == category]['amount'].sum()

                # Historical average (last 60 days excluding current month)
                hist_start = current_month_start - timedelta(days=60)
                hist_df = cat_df[(cat_df['date'] >= hist_start) & (cat_df['date'] < current_month_start)]

                if len(hist_df) > 0:
                    # Calculate daily average from history
                    days_in_history = (current_month_start - hist_start).days
                    daily_avg = hist_df['amount'].sum() / days_in_history

                    # Forecast for full month
                    forecasted = daily_avg * days_in_full_month

                    # Determine trend
                    if current_month_cat > forecasted * 1.2:
                        trend = "INCREASING"
                        confidence = "MEDIUM"
                    elif current_month_cat < forecasted * 0.8:
                        trend = "DECREASING"
                        confidence = "MEDIUM"
                    else:
                        trend = "STABLE"
                        confidence = "HIGH"
                else:
                    # No history, just project current month
                    forecasted = (current_month_cat / days_in_current_month) * days_in_full_month
                    trend = "STABLE"
                    confidence = "LOW"

                forecasts.append({
                    'category': category,
                    'current_month_spending': float(current_month_cat),
                    'forecasted_next_month': float(forecasted),
                    'confidence': confidence,
                    'trend': trend
                })

            logger.info(f"Generated forecasts for {len(forecasts)} categories")
            return forecasts

        except Exception as e:
            logger.error(f"Error forecasting spending: {e}")
            return []

    def generate_recommendations(self) -> List[Dict]:
        """
        Generate personalized savings and optimization recommendations

        Returns:
            List of recommendations sorted by priority
        """
        try:
            recommendations = []

            # Analyze spending patterns
            patterns = self.analyze_spending_patterns()
            category_spending = {p['category']: p for p in patterns['spending_by_category']}

            # Total spending
            total_spending = self.df['amount'].sum()

            # 1. High food delivery spending
            food_dining = category_spending.get('Food & Dining', {})
            if food_dining and food_dining['transaction_count'] > 15:
                avg_order = food_dining['average_amount']
                potential_savings = food_dining['transaction_count'] * 0.3 * avg_order  # Save 30% by cooking
                recommendations.append({
                    'type': 'savings',
                    'category': 'Food & Dining',
                    'title': 'Reduce Food Delivery Orders',
                    'message': f"You ordered food {food_dining['transaction_count']} times. "
                               f"Cooking 5 more meals at home could save you ₹{potential_savings:.0f}/month.",
                    'potential_savings': float(potential_savings),
                    'priority': 'HIGH',
                    'actionable': True
                })

            # 2. Weekend overspending
            if 'weekend_vs_weekday' in patterns and patterns['weekend_vs_weekday']:
                diff_pct = patterns['weekend_vs_weekday'].get('difference_pct', 0)
                if diff_pct > 30:
                    weekend_avg = patterns['weekend_vs_weekday']['weekend_avg']
                    weekday_avg = patterns['weekend_vs_weekday']['weekday_avg']
                    recommendations.append({
                        'type': 'info',
                        'category': 'General',
                        'title': 'Weekend Spending Pattern',
                        'message': f"You spend {diff_pct:.0f}% more on weekends (₹{weekend_avg:.0f} vs ₹{weekday_avg:.0f}). "
                                   f"Consider planning weekend activities with a budget.",
                        'potential_savings': None,
                        'priority': 'MEDIUM',
                        'actionable': True
                    })

            # 3. High category concentration
            top_category = patterns['spending_by_category'][0] if patterns['spending_by_category'] else None
            if top_category and top_category['percentage_of_total'] > 40:
                recommendations.append({
                    'type': 'alert',
                    'category': top_category['category'],
                    'title': f"High {top_category['category']} Spending",
                    'message': f"{top_category['category']} accounts for {top_category['percentage_of_total']:.0f}% "
                               f"(₹{top_category['total_amount']:.0f}) of your spending. Consider setting a budget limit.",
                    'potential_savings': None,
                    'priority': 'HIGH',
                    'actionable': True
                })

            # 4. Entertainment subscriptions
            entertainment = category_spending.get('Entertainment', {})
            if entertainment and entertainment['average_amount'] > 500:
                recommendations.append({
                    'type': 'optimization',
                    'category': 'Entertainment',
                    'title': 'Review Entertainment Subscriptions',
                    'message': f"Your average entertainment transaction is ₹{entertainment['average_amount']:.0f}. "
                               f"Review unused subscriptions to save money.",
                    'potential_savings': entertainment['average_amount'] * 0.3,
                    'priority': 'MEDIUM',
                    'actionable': True
                })

            # 5. Fuel/Transportation costs
            transportation = category_spending.get('Transportation', {})
            fuel = category_spending.get('Fuel', {})

            total_commute = (transportation.get('total_amount', 0) + fuel.get('total_amount', 0))
            if total_commute > 3000:
                recommendations.append({
                    'type': 'optimization',
                    'category': 'Transportation',
                    'title': 'High Commute Costs',
                    'message': f"You're spending ₹{total_commute:.0f}/month on transportation. "
                               f"Consider carpooling, public transport, or work-from-home options.",
                    'potential_savings': total_commute * 0.2,
                    'priority': 'MEDIUM',
                    'actionable': True
                })

            # 6. Groceries bulk buying opportunity
            groceries = category_spending.get('Groceries', {})
            if groceries and groceries['transaction_count'] > 20:
                recommendations.append({
                    'type': 'savings',
                    'category': 'Groceries',
                    'title': 'Bulk Buy Groceries',
                    'message': f"You shop for groceries {groceries['transaction_count']} times/month. "
                               f"Buying in bulk weekly could save time and ₹{groceries['total_amount'] * 0.1:.0f}.",
                    'potential_savings': groceries['total_amount'] * 0.1,
                    'priority': 'LOW',
                    'actionable': True
                })

            # 7. Overall budget recommendation
            if total_spending > 30000:
                recommendations.append({
                    'type': 'alert',
                    'category': 'General',
                    'title': 'Set Monthly Budget',
                    'message': f"Your total spending is ₹{total_spending:.0f}. "
                               f"Set category-wise budgets to track and control expenses.",
                    'potential_savings': None,
                    'priority': 'HIGH',
                    'actionable': True
                })

            # Sort by priority
            priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            recommendations = sorted(recommendations, key=lambda x: priority_order[x['priority']])

            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_spending': float(self.df['amount'].sum()),
            'total_transactions': int(len(self.df)),
            'average_transaction': float(self.df['amount'].mean()),
            'date_range': {
                'start': self.df['date'].min().strftime('%Y-%m-%d'),
                'end': self.df['date'].max().strftime('%Y-%m-%d')
            },
            'unique_categories': int(self.df['category'].nunique()),
            'categories': sorted(self.df['category'].unique().tolist())
        }

    def get_full_insights(self) -> Dict:
        """
        Get complete insights analysis

        Returns:
            Dict with all insights data
        """
        try:
            logger.info("Generating full insights analysis...")

            summary = self.get_summary()
            anomalies = self.detect_anomalies()
            patterns = self.analyze_spending_patterns()
            forecasts = self.forecast_spending()
            recommendations = self.generate_recommendations()

            return {
                'summary': summary,
                'anomalies': anomalies,
                'spending_by_category': patterns['spending_by_category'],
                'spending_by_day': patterns['spending_by_day'],
                'forecasts': forecasts,
                'recommendations': recommendations,
                'analysis_period': summary['date_range'],
                'total_transactions': summary['total_transactions'],
                'total_spending': summary['total_spending']
            }

        except Exception as e:
            logger.error(f"Error generating full insights: {e}")
            raise
