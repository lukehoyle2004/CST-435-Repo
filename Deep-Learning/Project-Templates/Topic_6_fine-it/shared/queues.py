"""The routing contract: which support queue owns each Banking77 intent.

FineTuneFleet's customer is the support desk of a digital bank. Every incoming
message has to land in one of eight queues, each staffed by a different team.
Banking77 labels each message with one of 77 fine-grained intents; this module
is the single, reviewable mapping from those intents to the eight queues.

The mapping rules, applied in this order so every intent has exactly one owner:

1. Anything the customer did not authorise, or a lost/stolen/compromised card or
   phone, goes to ``security_fraud`` -- the one queue where a slow answer costs
   the customer money.
2. Any question about a fee, a charge, or an exchange rate goes to ``fees_fx``,
   whatever product it was charged on (the fees team owns the price list).
3. Everything else goes to the team that owns the product: cards, transfers,
   top-ups, cash/ATM, card payments and refunds, or account and identity.

The queue descriptions are also what the zero-shot baseline sees, so they are
written to be fair to it: plain language, one sentence, no intent names.
"""
from __future__ import annotations

QUEUES: dict[str, str] = {
    "card_services": "ordering, delivering, activating, linking or using a physical or virtual card, its PIN and contactless",
    "transfers": "bank transfers into or out of the account, beneficiaries, and transfers that are pending, failed or missing",
    "top_ups": "adding money to the account by card, bank transfer, cash or cheque, and top-ups that are pending, failed or reverted",
    "cash_atm": "ATMs and cash withdrawals that were declined, pending, swallowed the card or gave the wrong amount",
    "fees_fx": "fees, extra charges, exchange rates and currency exchange",
    "payments_disputes": "card payments that were declined, pending, reverted or charged twice, and refunds",
    "security_fraud": "lost, stolen or compromised cards or phones, and payments, withdrawals or direct debits the customer does not recognise",
    "account_identity": "opening, closing or editing the account, identity verification, passcodes, and which countries, ages and cards are supported",
}

QUEUE_NAMES: list[str] = list(QUEUES)
QUEUE_TO_ID: dict[str, int] = {q: i for i, q in enumerate(QUEUE_NAMES)}

INTENT_TO_QUEUE: dict[str, str] = {
    # security_fraud -- rule 1
    "compromised_card": "security_fraud",
    "lost_or_stolen_card": "security_fraud",
    "lost_or_stolen_phone": "security_fraud",
    "card_payment_not_recognised": "security_fraud",
    "cash_withdrawal_not_recognised": "security_fraud",
    "direct_debit_payment_not_recognised": "security_fraud",
    # fees_fx -- rule 2
    "card_payment_fee_charged": "fees_fx",
    "card_payment_wrong_exchange_rate": "fees_fx",
    "cash_withdrawal_charge": "fees_fx",
    "exchange_charge": "fees_fx",
    "exchange_rate": "fees_fx",
    "exchange_via_app": "fees_fx",
    "extra_charge_on_statement": "fees_fx",
    "fiat_currency_support": "fees_fx",
    "top_up_by_bank_transfer_charge": "fees_fx",
    "top_up_by_card_charge": "fees_fx",
    "transfer_fee_charged": "fees_fx",
    "wrong_exchange_rate_for_cash_withdrawal": "fees_fx",
    # card_services
    "activate_my_card": "card_services",
    "apple_pay_or_google_pay": "card_services",
    "card_about_to_expire": "card_services",
    "card_acceptance": "card_services",
    "card_arrival": "card_services",
    "card_delivery_estimate": "card_services",
    "card_linking": "card_services",
    "card_not_working": "card_services",
    "change_pin": "card_services",
    "contactless_not_working": "card_services",
    "disposable_card_limits": "card_services",
    "get_disposable_virtual_card": "card_services",
    "get_physical_card": "card_services",
    "getting_spare_card": "card_services",
    "getting_virtual_card": "card_services",
    "order_physical_card": "card_services",
    "pin_blocked": "card_services",
    "virtual_card_not_working": "card_services",
    "visa_or_mastercard": "card_services",
    # transfers
    "balance_not_updated_after_bank_transfer": "transfers",
    "beneficiary_not_allowed": "transfers",
    "cancel_transfer": "transfers",
    "declined_transfer": "transfers",
    "failed_transfer": "transfers",
    "pending_transfer": "transfers",
    "receiving_money": "transfers",
    "transfer_into_account": "transfers",
    "transfer_not_received_by_recipient": "transfers",
    "transfer_timing": "transfers",
    # top_ups
    "automatic_top_up": "top_ups",
    "balance_not_updated_after_cheque_or_cash_deposit": "top_ups",
    "pending_top_up": "top_ups",
    "top_up_by_cash_or_cheque": "top_ups",
    "top_up_failed": "top_ups",
    "top_up_limits": "top_ups",
    "top_up_reverted": "top_ups",
    "topping_up_by_card": "top_ups",
    "verify_top_up": "top_ups",
    # cash_atm
    "atm_support": "cash_atm",
    "card_swallowed": "cash_atm",
    "declined_cash_withdrawal": "cash_atm",
    "pending_cash_withdrawal": "cash_atm",
    "wrong_amount_of_cash_received": "cash_atm",
    # payments_disputes
    "declined_card_payment": "payments_disputes",
    "pending_card_payment": "payments_disputes",
    "reverted_card_payment?": "payments_disputes",
    "request_refund": "payments_disputes",
    "Refund_not_showing_up": "payments_disputes",
    "transaction_charged_twice": "payments_disputes",
    # account_identity
    "age_limit": "account_identity",
    "country_support": "account_identity",
    "edit_personal_details": "account_identity",
    "passcode_forgotten": "account_identity",
    "supported_cards_and_currencies": "account_identity",
    "terminate_account": "account_identity",
    "unable_to_verify_identity": "account_identity",
    "verify_my_identity": "account_identity",
    "verify_source_of_funds": "account_identity",
    "why_verify_identity": "account_identity",
}

# The queue whose misses matter most; the service and the docs report its recall
# on its own, and low-confidence predictions are flagged for a human (see api/).
PRIORITY_QUEUE = "security_fraud"
