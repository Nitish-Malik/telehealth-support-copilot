#!/usr/bin/env python3
"""
Synthetic data for a Support Copilot (RAG) project.

Generates two files:

  policies.csv : doc_id, title, content, category
                 Hand-written knowledge-base articles (the documents the LLM retrieves).

  tickets.csv  : ticket_id, created_at, customer_id, channel, category, priority,
                 subject, description, order_id, status, resolution,
                 resolution_time_hours, related_doc_id
                 Customer tickets. `related_doc_id` is the ground-truth policy that
                 answers the ticket, so you can measure retrieval (hit-rate / MRR)
                 and answer quality (compare model output with `resolution`).

Standard library only. Reproducible via --seed.

Usage (from the project root):
    python src/generate_data.py
    python src/generate_data.py --n-tickets 5000 --noise 0.25 --out-dir data --seed 7
"""

import argparse
import csv
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. POLICY / KNOWLEDGE-BASE DOCUMENTS
#    (doc_id, title, category, content)
#    Numbers are deliberately concrete so answers can be checked against them.
# ----------------------------------------------------------------------------
POLICIES = [
    ("DOC-001", "Standard Refund Policy", "Billing",
     "Approved refunds are credited to the original payment method within 5-7 business days. "
     "UPI and wallet refunds typically reach the customer in 1-3 business days, while credit and "
     "debit card refunds can take up to 7 business days depending on the bank. Cash on delivery "
     "orders are refunded to the customer's bank account or to the store wallet, whichever the "
     "customer chooses. Return requests are reviewed within 48 hours of the returned item being "
     "received at our warehouse. Shipping fees are refunded only if the return was caused by our error."),

    ("DOC-002", "Return Policy for Non-Medicine Products", "Returns & Refunds",
     "Wellness devices, supplements, personal care and other non-medicine products can be returned "
     "within 7 days of delivery. The item must be unused, unopened and in its original packaging "
     "with all tags and accessories. Opened supplements, personal care items and consumables cannot "
     "be returned for hygiene reasons unless they arrived damaged or incorrect. Return pickup is "
     "arranged free of cost for eligible returns, and the refund follows the Standard Refund Policy."),

    ("DOC-003", "Medicine Return Policy", "Returns & Refunds",
     "Medicines cannot be returned once delivered, except when the wrong medicine was sent, the "
     "package arrived damaged or tampered with, or the medicine is expired or near expiry. Such "
     "issues must be reported within 48 hours of delivery with clear photos of the product, batch "
     "number and outer packaging. After verification, we either send the correct medicine free of "
     "cost or issue a full refund, as the customer prefers. Medicines kept for more than 48 hours "
     "without a report are not eligible for return."),

    ("DOC-004", "Order Cancellation and Address Changes", "Orders",
     "Orders can be cancelled free of charge any time before they are dispatched, from the Orders "
     "section of the app or by contacting support. Once an order has been dispatched it cannot be "
     "cancelled, but the customer may refuse delivery and the refund will be issued after the "
     "package returns to us. Orders awaiting prescription verification can be cancelled at any time "
     "before verification is completed. Any coupon used is restored if the order is cancelled before "
     "dispatch. The delivery address can be edited until the order is packed; after that, the "
     "customer should cancel before dispatch and place a fresh order."),

    ("DOC-005", "Standard Delivery Terms", "Delivery",
     "Standard delivery takes 2-4 business days in metro cities and 4-7 business days in other "
     "serviceable locations. Delivery is free on orders of Rs 499 and above, and a delivery fee of "
     "Rs 49 applies to smaller orders. Delivery is attempted up to three times, after which the order "
     "is returned to us and refunded. A reachable phone number is required for the delivery partner, "
     "and an OTP is shared at delivery to confirm the handover."),

    ("DOC-006", "Express Same-Day Delivery", "Delivery",
     "Express same-day delivery is available in select cities for orders placed before 2 PM, with an "
     "additional fee of Rs 99. Orders placed after 2 PM are delivered the next day. Express delivery "
     "is not available for items that need prescription verification beyond the cut-off time. If an "
     "express order is delayed beyond the promised slot, the express fee is refunded."),

    ("DOC-007", "Delayed or Missing Delivery", "Delivery",
     "If an order is delayed by more than 3 days beyond the estimated delivery date, the customer may "
     "cancel it for a full refund. Delays of more than 5 days also earn Rs 50 of store wallet credit. "
     "If tracking shows delivered but the customer has not received the package, it must be reported "
     "within 24 hours and we investigate with the delivery partner within 48 hours. Confirmed lost "
     "shipments are reshipped at no cost or fully refunded, as the customer prefers."),

    ("DOC-008", "Prescription Verification Rules", "Prescription",
     "Schedule H and H1 medicines require a valid prescription. A valid prescription must show the "
     "doctor's name and registration number, the patient's name, the date of issue, and the medicine "
     "names with dosage. Prescriptions for long-term medicines are accepted if not older than 6 "
     "months; for all other medicines the prescription must be issued within the last 30 days. "
     "Uploaded prescriptions are reviewed by a licensed pharmacist within 4 business hours, the "
     "customer may be called to confirm details, and the order is dispatched only after verification."),

    ("DOC-009", "Rejected Prescriptions", "Prescription",
     "A prescription may be rejected if it is blurry or cropped, is missing the doctor's registration "
     "number, is out of date, does not match the patient's name, or lists a medicine that cannot be "
     "sold online. The customer is notified in the app and by SMS with the reason. A corrected "
     "prescription can be re-uploaded against the same order within 24 hours; otherwise the order is "
     "cancelled and any prepaid amount is refunded in 5-7 business days. Customers without a valid "
     "prescription can book a teleconsultation to get one."),

    ("DOC-010", "Controlled and Restricted Medicines", "Prescription",
     "Schedule X and narcotic medicines are not sold on the platform under any circumstances. "
     "Schedule H1 medicines, such as certain antibiotics and sedatives, need a valid prescription and "
     "the pharmacist records each sale as required by law. Quantities are limited to what the "
     "prescription states, and refills beyond the prescribed quantity are not allowed."),

    ("DOC-011", "Teleconsultation Policy", "Consultation",
     "Teleconsultations with registered doctors last up to 15 minutes. A follow-up with the same "
     "doctor within 7 days of the first consultation is free. Appointments can be rescheduled or "
     "cancelled up to 2 hours before the slot at no charge. When clinically appropriate, a "
     "prescription is shared within 30 minutes of the consultation ending. Doctors do not treat "
     "medical emergencies; in an emergency, the customer should visit the nearest hospital."),

    ("DOC-012", "Consultation Refund Policy", "Billing",
     "A consultation is refunded in full if the doctor does not join within 30 minutes of the slot, "
     "if the call fails because of a technical issue on our side, or if the customer cancels at least "
     "2 hours before the appointment. Consultation fees are not refunded when the customer is a "
     "no-show or when the consultation was completed. Refunds follow the Standard Refund Policy timelines."),

    ("DOC-013", "Accepted Payment Methods", "Billing",
     "We accept UPI, credit and debit cards, net banking, popular wallets and cash on delivery. Cash "
     "on delivery is available for orders up to Rs 5,000 and is not available with express delivery. "
     "Card details are handled by PCI-compliant payment partners and are never stored by us. EMI on "
     "cards is available for orders above Rs 3,000."),

    ("DOC-014", "Failed Payments and Double Charges", "Billing",
     "If money is deducted but the order is not confirmed, the amount is automatically reversed "
     "within 5-7 business days. Duplicate charges are refunded automatically within the same period. "
     "If the reversal has not arrived after 7 business days, the customer should share the transaction "
     "ID and a bank statement screenshot so the team can raise it with the payment partner, which "
     "typically resolves it within 3 business days."),

    ("DOC-015", "Coupons and Offers", "Billing",
     "Only one coupon can be applied per order, and coupons cannot be combined with other discount "
     "offers unless stated. Each coupon has a minimum order value and an expiry date shown in its "
     "terms. Coupons do not apply to prescription medicines unless specifically mentioned. If an order "
     "is cancelled before dispatch the coupon is restored; coupons are not restored after dispatch or "
     "for completed returns."),

    ("DOC-016", "Account Access and Security", "Account",
     "Passwords can be reset using an OTP sent to the registered mobile number or email. After 5 "
     "failed login attempts the account is locked for 30 minutes. Customers who no longer have access "
     "to their registered mobile number can verify their identity with a recent order ID and email to "
     "update it. We never ask for OTPs, card PINs or passwords over call or chat."),

    ("DOC-017", "Data Privacy and Deletion", "Privacy",
     "Health records, prescriptions and order history are encrypted and shared only with the licensed "
     "pharmacists and doctors involved in the customer's care. Customers can request a copy of their "
     "data or ask for account deletion from the app settings or through support. Deletion requests are "
     "completed within 30 days, except for records we are legally required to retain, such as "
     "prescription sale records. Customer data is not sold to third parties."),

    ("DOC-018", "Product Authenticity and Storage", "Product Safety",
     "All products are sourced directly from licensed manufacturers and authorised distributors. "
     "Temperature-sensitive medicines such as insulin ship in cold-chain packaging and must be placed "
     "in a refrigerator on receipt. Customers who suspect a product is counterfeit or tampered with "
     "should stop using it, keep the packaging, and report it within 48 hours with photos and the "
     "batch number for investigation and a full refund."),

    ("DOC-019", "Expiry and Shelf Life Policy", "Product Safety",
     "Medicines are delivered with at least 3 months of remaining shelf life, and supplements with at "
     "least 6 months, unless the product page states otherwise. If a product arrives with a shorter "
     "shelf life, the customer can request a replacement or a refund within 48 hours of delivery. "
     "Expiry is counted to the last day of the month printed on the pack."),

    ("DOC-020", "Adverse Reaction Reporting", "Product Safety",
     "Customers who experience an allergic reaction or side effect after taking a medicine should stop "
     "using it and consult a doctor immediately; in a severe reaction they should seek emergency care. "
     "The reaction can be reported to support within 24 hours with the medicine name and batch number, "
     "and the case is forwarded to our pharmacovigilance team. Support agents cannot give medical "
     "advice, and medicines that have been consumed are not returned; the report is used for safety follow-up."),

    ("DOC-021", "Support Response Times and Escalation", "Support",
     "First response times are 4 hours on chat and phone during working hours (8 AM to 10 PM) and 24 "
     "hours on email. Complex cases are resolved within 3 business days. If a ticket is unresolved "
     "after 3 business days, the customer can ask for escalation to a senior agent, and after a further "
     "48 hours to the Grievance Officer, whose details are on the Contact Us page."),
]

# ----------------------------------------------------------------------------
# 2. TICKET SCENARIOS
#    Each scenario is tied to one policy (ground truth) and carries a reference
#    resolution that is consistent with that policy's content.
#    Placeholders: {order_id} {amount} {amount_low} {medicine} {device} {txn_id}
# ----------------------------------------------------------------------------
SCENARIOS = []


def S(doc_id, subjects, bodies, resolution, priority="medium"):
    SCENARIOS.append(dict(doc=doc_id, subjects=subjects, bodies=bodies,
                          resolution=resolution, priority=priority))


# --- DOC-001 Standard Refund Policy
S("DOC-001",
  ["Refund not received", "Where is my refund?", "Refund status for {order_id}"],
  ["My refund of Rs {amount} for order {order_id} was approved but I have not received the money. How long does it take?",
   "I returned order {order_id} last week and was told Rs {amount} would be refunded. Nothing in my account yet."],
  "Approved refunds are credited to the original payment method within 5-7 business days. UPI and wallet refunds usually "
  "arrive in 1-3 business days, and card refunds can take up to 7 business days depending on your bank. If it has been more "
  "than 7 business days since approval, please share your bank reference and we will escalate it.")
S("DOC-001",
  ["Refund for cash on delivery order", "COD refund how?"],
  ["I paid cash on delivery for {order_id} and returned it. How will I get my Rs {amount} back?",
   "Order {order_id} was COD. Where will the refund go?"],
  "Cash on delivery orders are refunded either to your bank account or to your store wallet, whichever you choose. Share your "
  "bank details or select the wallet option, and the refund will be processed within 5-7 business days after the return is approved.")

# --- DOC-002 Return Policy for Non-Medicine Products
S("DOC-002",
  ["Return a {device}", "Want to return my {device}", "{device} return request"],
  ["I bought a {device} in order {order_id} a few days ago and it is not what I expected. It is unused and still in the box. Can I return it?",
   "Please arrange a return for the {device} from {order_id}. Never opened it."],
  "Yes. Non-medicine products can be returned within 7 days of delivery if they are unused and in the original packaging with all "
  "accessories. Return pickup is free, and the refund follows the standard 5-7 business day timeline once the return is approved.")
S("DOC-002",
  ["Return after 10 days", "Late return of {device}"],
  ["I received the {device} (order {order_id}) 10 days ago and want to return it. It is unopened.",
   "Is it too late to return the {device} from {order_id}? Delivered 10 days back."],
  "Non-medicine products can be returned only within 7 days of delivery, so a standard return is no longer possible for an order "
  "delivered 10 days ago. If the item arrived damaged or incorrect, please share photos and we will review it.")
S("DOC-002",
  ["Return opened protein powder", "Opened supplement return"],
  ["I opened the protein powder from order {order_id} and did not like the taste. Can I return it?",
   "Opened my supplement jar from {order_id}, giving me stomach discomfort with the flavour. Want to return."],
  "Opened supplements and other consumables cannot be returned for hygiene reasons, unless they arrived damaged or incorrect. "
  "Since the product is opened and was delivered as ordered, it is not eligible for return.")

# --- DOC-003 Medicine Return Policy
S("DOC-003",
  ["Wrong medicine delivered", "Received different medicine", "Wrong item in {order_id}"],
  ["I ordered {medicine} but received a different medicine in order {order_id}. Please fix this.",
   "The medicine in my package for {order_id} does not match what I ordered ({medicine})."],
  "Sorry for the mix-up. Wrong-medicine issues must be reported within 48 hours of delivery with photos of the product, batch "
  "number and outer packaging. Please upload these and, after verification, we will send the correct medicine free of cost or "
  "issue a full refund, whichever you prefer.", priority="high")
S("DOC-003",
  ["Return extra medicine", "Ordered twice by mistake"],
  ["I ordered {medicine} twice by mistake in {order_id}. Can I return the extra box?",
   "I no longer need the {medicine} from {order_id}. Please arrange a return."],
  "Medicines cannot be returned once delivered unless the wrong medicine was sent, the package was damaged or tampered with, or "
  "the medicine is expired or near expiry. An accidental duplicate or a change of mind is not eligible for return.")
S("DOC-003",
  ["Damaged medicine package", "Seal broken on medicine box"],
  ["The box of {medicine} in order {order_id} arrived crushed and the seal was torn.",
   "Package for {order_id} came damaged. The {medicine} strip looks tampered with."],
  "Damaged or tampered packages are eligible for a free replacement or a full refund. Please report it within 48 hours of delivery "
  "with clear photos of the product, batch number and outer packaging, and we will verify and resolve it.", priority="high")

# --- DOC-004 Order Cancellation and Address Changes
S("DOC-004",
  ["Cancel my order", "Please cancel {order_id}"],
  ["Please cancel order {order_id}, I placed it by mistake. It still shows as being packed.",
   "I want to cancel {order_id}. It has not been shipped yet."],
  "Orders can be cancelled free of charge before they are dispatched. Since your order has not been dispatched, it can be cancelled "
  "now. Any coupon you used will be restored, and the refund will follow the standard 5-7 business day timeline.")
S("DOC-004",
  ["Cancel order that is out for delivery", "Cancellation after dispatch"],
  ["Order {order_id} is already out for delivery but I want to cancel it.",
   "Can I cancel {order_id}? Tracking says it has been dispatched."],
  "Once an order has been dispatched it cannot be cancelled. You can refuse delivery when it arrives, and the refund will be issued "
  "after the package returns to us.")
S("DOC-004",
  ["Cancel order awaiting prescription check", "Prescription under verification, cancel?"],
  ["My prescription for order {order_id} is still under verification. Can I cancel?",
   "I uploaded a prescription for {order_id} but changed my mind. Can I cancel while it is being verified?"],
  "Yes. Orders awaiting prescription verification can be cancelled at any time before verification is completed. Prepaid amounts are "
  "refunded in 5-7 business days.")
S("DOC-004",
  ["Change delivery address", "Wrong address on order {order_id}"],
  ["I entered the wrong address for order {order_id}. Can I change it?",
   "Need to update the delivery address on {order_id} before it ships."],
  "The delivery address can be edited from the Orders section until the order is packed. If it is already packed, please cancel the "
  "order before it is dispatched and place a fresh one with the correct address.")

# --- DOC-005 Standard Delivery Terms
S("DOC-005",
  ["Delivery fee charged", "Why was I charged for delivery?"],
  ["Why was I charged a delivery fee on order {order_id}? The order value was only Rs {amount_low}.",
   "Delivery charge of Rs 49 added to {order_id} (Rs {amount_low}). Is that correct?"],
  "A delivery fee of Rs 49 applies to orders below Rs 499, and delivery is free on orders of Rs 499 and above. Adding items to "
  "reach Rs 499 will make the delivery free.", priority="low")
S("DOC-005",
  ["How long will delivery take?", "Delivery time to my town"],
  ["I live in a small town and placed order {order_id} yesterday. How many days will delivery take?",
   "Estimated delivery date for {order_id}? I am not in a metro city."],
  "Standard delivery takes 2-4 business days in metro cities and 4-7 business days in other serviceable locations, so expect "
  "your order within 4-7 business days.", priority="low")
S("DOC-005",
  ["Delivery attempted once only", "Order returned without delivery"],
  ["The delivery partner called once and then marked order {order_id} as returned. Why?",
   "My order {order_id} went back to the warehouse after one missed call."],
  "Delivery is attempted up to three times, and the order is returned to us only after that, with a refund. Please make sure your "
  "phone number is reachable. We can reship the order or refund it, as you prefer.")
S("DOC-005",
  ["OTP asked at delivery", "Is delivery OTP safe?"],
  ["The delivery partner is asking for an OTP for order {order_id}. Is it safe to share?",
   "I got an OTP message for {order_id}. Should I give it to the delivery person?"],
  "Yes. An OTP is shared at the time of delivery to confirm the handover. Share it only with the delivery partner who is handing "
  "over your package.", priority="low")

# --- DOC-006 Express Same-Day Delivery
S("DOC-006",
  ["Express order not delivered today", "Same-day delivery missed"],
  ["I placed an express order {order_id} at 4 PM and it was not delivered today.",
   "Paid Rs 99 for same-day delivery on {order_id} but it did not arrive. I ordered around 5 PM."],
  "Express same-day delivery applies to orders placed before 2 PM, and orders placed after 2 PM are delivered the next day. "
  "Your order will be delivered tomorrow.")
S("DOC-006",
  ["Express delivery was late", "Refund express fee"],
  ["My express order {order_id} was promised by 6 PM and arrived at 10 PM. Can I get the express fee back?",
   "Express delivery on {order_id} missed the promised slot. I want the Rs 99 refunded."],
  "If an express order is delayed beyond the promised slot, the express fee of Rs 99 is refunded. We have initiated the refund and "
  "it will follow the standard 5-7 business day timeline.")

# --- DOC-007 Delayed or Missing Delivery
S("DOC-007",
  ["Order is very late", "Delayed delivery {order_id}"],
  ["My order {order_id} is now more than 6 days past the estimated delivery date.",
   "Order {order_id} is way behind schedule. What can I do?"],
  "If an order is delayed by more than 3 days beyond the estimated delivery date, you can cancel it for a full refund. Since the "
  "delay is more than 5 days, you are also eligible for Rs 50 of store wallet credit.", priority="high")
S("DOC-007",
  ["Marked delivered but not received", "Package not received"],
  ["Tracking shows order {order_id} as delivered but I never received it.",
   "{order_id} says delivered yesterday. Nothing came."],
  "Please report non-receipt within 24 hours of the delivered status. We will investigate with the delivery partner within 48 "
  "hours, and if the shipment is confirmed lost we will reship it at no cost or refund it in full, as you prefer.", priority="high")
S("DOC-007",
  ["Tracking not updating", "Shipment stuck"],
  ["The tracking for {order_id} has not changed for a week. I think it is lost.",
   "Order {order_id} seems stuck in transit with no update."],
  "We will check with the delivery partner. If the shipment is confirmed lost, it will be reshipped at no cost or fully refunded, "
  "whichever you prefer.")

# --- DOC-008 Prescription Verification Rules
S("DOC-008",
  ["What should a prescription include?", "Prescription requirements"],
  ["What details must my prescription have for ordering {medicine}?",
   "Which information is needed on a prescription for {medicine}?"],
  "A valid prescription must show the doctor's name and registration number, the patient's name, the date of issue, and the medicine "
  "names with dosage.", priority="low")
S("DOC-008",
  ["How long does prescription verification take?", "Verification delay for {order_id}"],
  ["I uploaded my prescription for {order_id}. How long will verification take?",
   "Prescription for {order_id} has been under review for a while. What is the timeline?"],
  "Uploaded prescriptions are reviewed by a licensed pharmacist within 4 business hours. We may call you to confirm details, and "
  "the order is dispatched only after verification.", priority="low")
S("DOC-008",
  ["Is a 4-month-old prescription valid?", "Old prescription for long-term medicine"],
  ["I take {medicine} long term and my prescription is 4 months old. Will it be accepted?",
   "Can I order {medicine} on a prescription from 4 months ago?"],
  "Prescriptions for long-term medicines are accepted if they are not older than 6 months, so a 4-month-old prescription is valid "
  "provided it shows the doctor's name and registration number, the patient's name and the dosage. For other medicines the "
  "prescription must be issued within the last 30 days.")

# --- DOC-009 Rejected Prescriptions
S("DOC-009",
  ["Prescription rejected", "Why was my prescription rejected?"],
  ["My prescription for order {order_id} was rejected. What went wrong?",
   "Got a message that my prescription is invalid for {order_id}."],
  "A prescription can be rejected if it is blurry or cropped, missing the doctor's registration number, out of date, not matching "
  "the patient's name, or listing a medicine that cannot be sold online. The reason is shown in the app and sent by SMS. You can "
  "re-upload a corrected prescription against the same order within 24 hours.")
S("DOC-009",
  ["Refund for rejected prescription order", "Cancelled after prescription rejection"],
  ["I could not fix my prescription in time and order {order_id} was cancelled. When do I get my Rs {amount} back?",
   "Prescription rejected and {order_id} cancelled. Where is my money?"],
  "If a corrected prescription is not uploaded within 24 hours, the order is cancelled and any prepaid amount is refunded within 5-7 "
  "business days.")
S("DOC-009",
  ["No prescription, what now?", "How to get a prescription"],
  ["I do not have a prescription for {medicine}. How can I order it?",
   "Need {medicine} but have no valid prescription. Options?"],
  "You can book a teleconsultation with a registered doctor on the platform, who can issue a prescription if it is clinically "
  "appropriate. You can then upload it and place your order.", priority="low")

# --- DOC-010 Controlled and Restricted Medicines
S("DOC-010",
  ["Order sleeping pills without prescription", "Alprazolam without prescription"],
  ["Can I buy alprazolam without a prescription? I do not have time to see a doctor.",
   "I want to order sedatives without a prescription. Is that possible?"],
  "Schedule H1 medicines need a valid prescription and cannot be sold without one, and Schedule X and narcotic medicines are not "
  "sold on the platform at all. You can book a teleconsultation, where the doctor will decide whether the medicine is appropriate.")
S("DOC-010",
  ["Order more tablets than prescribed", "Quantity limit on medicine"],
  ["My prescription says 10 tablets of {medicine} but I want 30. Why can't I order that?",
   "Cannot add more than the prescribed quantity of {medicine} to my cart."],
  "Quantities are limited to what the prescription states, and refills beyond the prescribed quantity are not allowed. For more, "
  "you would need an updated prescription, which you can get through a teleconsultation.", priority="low")

# --- DOC-011 Teleconsultation Policy
S("DOC-011",
  ["Consultation duration and follow-up", "Is follow-up free?"],
  ["How long is a doctor consultation, and do I have to pay again for a follow-up?",
   "I consulted a doctor 3 days ago. Is a follow-up with the same doctor free?"],
  "Teleconsultations last up to 15 minutes. A follow-up with the same doctor within 7 days of the first consultation is free.",
  priority="low")
S("DOC-011",
  ["Reschedule appointment", "Change doctor appointment time"],
  ["I need to reschedule my consultation tomorrow at 6 PM. Can I do it at 3 PM today?",
   "Want to cancel my consultation slot at 10 AM. It is 9 AM now."],
  "Appointments can be rescheduled or cancelled at no charge up to 2 hours before the slot. Changes within 2 hours of the "
  "appointment are not possible.", priority="low")
S("DOC-011",
  ["No prescription after consultation", "Prescription not received"],
  ["My consultation ended 20 minutes ago and I have not received a prescription.",
   "Doctor did not share any prescription after my consult."],
  "When clinically appropriate, a prescription is shared within 30 minutes of the consultation ending. Please check again after "
  "30 minutes. If the doctor decided that no medicine was needed, no prescription will be issued.")
S("DOC-011",
  ["Chest pain, need a doctor now", "Emergency consultation"],
  ["I have severe chest pain and difficulty breathing. Can I book a quick consultation?",
   "My father is unconscious. Can a doctor on the app help right now?"],
  "Doctors on the platform do not treat medical emergencies. Please go to the nearest hospital or call emergency services "
  "immediately.", priority="urgent")

# --- DOC-012 Consultation Refund Policy
S("DOC-012",
  ["Doctor did not join", "Doctor no-show for consultation"],
  ["My consultation was at 5 PM and the doctor never joined. I waited 40 minutes. I want a refund.",
   "Doctor did not join the call for my appointment. Please refund Rs {amount_low}."],
  "If the doctor does not join within 30 minutes of the slot, the consultation fee is refunded in full. We have initiated "
  "your refund, which follows the standard 5-7 business day timeline.", priority="high")
S("DOC-012",
  ["Refund for missed appointment", "I missed my consultation"],
  ["I missed my consultation by mistake. Can I get a refund?",
   "Could not attend my booked consultation. Please refund the fee."],
  "Consultation fees are not refunded when the customer is a no-show or when the consultation has been completed. A refund is "
  "possible only if you cancel at least 2 hours before the appointment.")
S("DOC-012",
  ["Call dropped during consultation", "Technical issue in consult"],
  ["The video call kept failing due to app errors and my consultation could not happen.",
   "Consult ended after 2 minutes because of a technical problem. I want my money back."],
  "If the call fails because of a technical issue on our side, the consultation is refunded in full. The refund follows the "
  "standard 5-7 business day timeline.")

# --- DOC-013 Accepted Payment Methods
S("DOC-013",
  ["COD for large order", "Cash on delivery not available"],
  ["I cannot select cash on delivery for my Rs 6,500 order. Why?",
   "COD option is missing at checkout for a high-value order."],
  "Cash on delivery is available only for orders up to Rs 5,000. For a larger order, please pay with UPI, a card, net banking "
  "or a wallet.", priority="low")
S("DOC-013",
  ["EMI option", "No EMI on my order"],
  ["I do not see an EMI option for my Rs 2,500 order. Why?",
   "Is EMI available for small orders?"],
  "EMI on cards is available for orders above Rs 3,000, so it is not offered on a Rs 2,500 order.", priority="low")
S("DOC-013",
  ["Do you store my card details?", "Card safety question"],
  ["Are my card details stored when I pay on the app?",
   "Is it safe to pay by card? Do you save my card number?"],
  "Card details are handled by PCI-compliant payment partners and are never stored by us.", priority="low")

# --- DOC-014 Failed Payments and Double Charges
S("DOC-014",
  ["Money deducted, order not placed", "Payment failed but amount debited"],
  ["Rs {amount} was deducted from my account (transaction {txn_id}) but I did not get an order confirmation.",
   "Payment showed failed, yet Rs {amount} left my bank account. Txn ID {txn_id}."],
  "If money is deducted and the order is not confirmed, the amount is automatically reversed within 5-7 business days. If it has "
  "not arrived after 7 business days, please share the transaction ID and a bank statement screenshot so we can raise it with the "
  "payment partner, which typically resolves it within 3 business days.", priority="high")
S("DOC-014",
  ["Charged twice", "Duplicate payment for {order_id}"],
  ["I was charged twice for order {order_id}. Transaction {txn_id}.",
   "Two debits of Rs {amount} for the same order {order_id}."],
  "Duplicate charges are refunded automatically within 5-7 business days. If the extra amount does not arrive after 7 business "
  "days, share the transaction ID and a bank statement screenshot and we will escalate it with the payment partner.", priority="high")

# --- DOC-015 Coupons and Offers
S("DOC-015",
  ["Cannot apply two coupons", "Combine coupons"],
  ["Why can't I use two coupons on order {order_id}?",
   "I want to stack my welcome coupon with a festival offer. It says not applicable."],
  "Only one coupon can be applied per order, and coupons cannot be combined with other discount offers unless the offer says so.",
  priority="low")
S("DOC-015",
  ["Coupon not applying to medicines", "Offer invalid for prescription items"],
  ["My coupon does not work on {medicine}. Is it broken?",
   "Coupon code rejected at checkout with a prescription medicine in the cart."],
  "Coupons do not apply to prescription medicines unless the offer specifically mentions them. Please check the coupon terms "
  "for its minimum order value and expiry as well.", priority="low")
S("DOC-015",
  ["Coupon lost after cancellation", "Coupon not restored"],
  ["I cancelled order {order_id} before it shipped but my coupon is gone.",
   "Coupon used on a cancelled order {order_id} was not returned to my account."],
  "A coupon is restored when an order is cancelled before dispatch. We have restored it to your account. Coupons are not "
  "restored after dispatch or for completed returns.", priority="low")

# --- DOC-016 Account Access and Security
S("DOC-016",
  ["Forgot password", "Cannot log in"],
  ["I forgot my password and cannot log in.",
   "Need to reset my password. How?"],
  "You can reset your password with an OTP sent to your registered mobile number or email, using the Forgot Password option.",
  priority="low")
S("DOC-016",
  ["Account locked", "Too many login attempts"],
  ["My account says locked after I typed the wrong password several times. When can I log in?",
   "Account locked since I tried logging in multiple times."],
  "After 5 failed login attempts the account is locked for 30 minutes. You can log in again after that, or reset your "
  "password using an OTP.", priority="low")
S("DOC-016",
  ["Lost my old phone number", "Update registered mobile"],
  ["I no longer have access to my registered mobile number. How can I change it?",
   "Old number is deactivated and I cannot receive OTP."],
  "You can verify your identity with a recent order ID and your email to update the registered mobile number.")
S("DOC-016",
  ["Caller asked for my OTP", "Suspicious call asking OTP"],
  ["Someone claiming to be from your company called and asked for my OTP. Is that genuine?",
   "I got a chat message asking for my card PIN to process a refund. Is it you?"],
  "No. We never ask for OTPs, card PINs or passwords over call or chat. Please do not share them with anyone, and report the "
  "number or message to us so we can look into it.", priority="urgent")

# --- DOC-017 Data Privacy and Deletion
S("DOC-017",
  ["Delete my account", "Account deletion request"],
  ["I want my account and data deleted. How long will it take?",
   "Please delete all my data from your platform."],
  "You can request deletion from the app settings or through support. Deletion requests are completed within 30 days, except for "
  "records we are legally required to retain, such as prescription sale records.")
S("DOC-017",
  ["Do you share my prescriptions?", "Is my health data shared?"],
  ["Who can see my uploaded prescriptions and order history?",
   "Do you share my health data with third parties or advertisers?"],
  "Health records, prescriptions and order history are encrypted and shared only with the licensed pharmacists and doctors "
  "involved in your care. Your data is not sold to third parties.", priority="low")
S("DOC-017",
  ["Copy of my data", "Request my data"],
  ["Can I get a copy of all the data you hold on me?",
   "I would like to download my order and prescription records."],
  "You can request a copy of your data from the app settings or through support.", priority="low")

# --- DOC-018 Product Authenticity and Storage
S("DOC-018",
  ["How to store insulin", "Storage for refrigerated medicine"],
  ["My insulin pens arrived in a cold pack. How should I store them?",
   "Do I need to refrigerate the medicine delivered in order {order_id}? It came in cold-chain packaging."],
  "Temperature-sensitive medicines such as insulin ship in cold-chain packaging and must be placed in a refrigerator as soon as "
  "you receive them.", priority="high")
S("DOC-018",
  ["Suspect fake medicine", "Product looks counterfeit"],
  ["The {medicine} I received looks different from what I usually buy. I think it may be fake.",
   "Packaging of {medicine} in {order_id} has spelling mistakes and the print is smudged."],
  "Please stop using the product, keep the packaging, and report it within 48 hours with photos and the batch number. We will "
  "investigate and issue a full refund. All our products are sourced from licensed manufacturers and authorised distributors.",
  priority="urgent")

# --- DOC-019 Expiry and Shelf Life Policy
S("DOC-019",
  ["Medicine near expiry", "Short expiry on delivered medicine"],
  ["The {medicine} I received in order {order_id} expires in 2 months.",
   "Delivered {medicine} has only 6 weeks left before expiry."],
  "Medicines are delivered with at least 3 months of remaining shelf life. You can request a replacement or a refund within 48 "
  "hours of delivery. Please share a photo of the expiry date and batch number.", priority="high")
S("DOC-019",
  ["Supplement expiry too soon", "Short shelf life on supplement"],
  ["My supplement from {order_id} has only 4 months of shelf life left. Is that acceptable?",
   "Vitamin bottle I received expires in 4 months."],
  "Supplements are delivered with at least 6 months of remaining shelf life, so a 4-month shelf life is below our standard. "
  "You can request a replacement or a refund within 48 hours of delivery.")

# --- DOC-020 Adverse Reaction Reporting
S("DOC-020",
  ["Rash after taking medicine", "Allergic reaction to {medicine}"],
  ["I developed a rash and itching after taking {medicine} from order {order_id}.",
   "Feeling dizzy and nauseous after starting {medicine}. Is the medicine faulty?"],
  "Please stop using the medicine and consult a doctor right away. If the reaction is severe, seek emergency care. You can report "
  "it to us within 24 hours with the medicine name and batch number, and we will forward it to our pharmacovigilance team. Our "
  "support agents cannot give medical advice.", priority="urgent")
S("DOC-020",
  ["Return medicine that caused side effects", "Refund for medicine with side effects"],
  ["The {medicine} gave me side effects. Can I return the rest and get a refund?",
   "Had a bad reaction to {medicine} from {order_id}. I want a refund."],
  "Medicines that have been consumed are not returned. Please stop using it and consult a doctor, and report the reaction to us "
  "with the medicine name and batch number so our pharmacovigilance team can follow up for safety.", priority="high")

# --- DOC-021 Support Response Times and Escalation
S("DOC-021",
  ["No reply to my email", "Waiting for support response"],
  ["I emailed support 2 days ago about {order_id} and have not heard back.",
   "No response on my email for over a day."],
  "First response on email is within 24 hours and on chat or phone within 4 hours during working hours (8 AM to 10 PM). We "
  "have flagged your email for a priority reply.")
S("DOC-021",
  ["Escalate my unresolved ticket", "Ticket open for 4 days"],
  ["My ticket about order {order_id} has been open for 4 days with no resolution. I want to escalate.",
   "Still no resolution after 4 days. Please escalate this to someone senior."],
  "Complex cases are resolved within 3 business days. Since your ticket is past that, we have escalated it to a senior agent. If "
  "it is still unresolved after a further 48 hours, you can reach the Grievance Officer, whose details are on the Contact Us "
  "page.", priority="high")

# ----------------------------------------------------------------------------
# 3. VALUE BANKS & HELPERS
# ----------------------------------------------------------------------------
MEDICINES = ["Paracetamol 500mg", "Metformin 500mg", "Atorvastatin 10mg", "Amoxicillin 500mg",
             "Cetirizine 10mg", "Pantoprazole 40mg", "Telmisartan 40mg", "Azithromycin 500mg",
             "Levothyroxine 50mcg", "Vitamin D3 60K", "Ibuprofen 400mg", "Montelukast 10mg"]
DEVICES = ["BP monitor", "glucometer", "digital thermometer", "pulse oximeter",
           "nebulizer", "weighing scale", "heating pad", "steam inhaler"]
CHANNELS = (["chat", "app", "email", "phone"], [0.40, 0.25, 0.25, 0.10])
STATUSES = (["resolved", "closed", "open", "escalated"], [0.68, 0.14, 0.10, 0.08])
PRIORITY_WEIGHTS = {
    "low":    {"low": 0.65, "medium": 0.35},
    "medium": {"low": 0.20, "medium": 0.60, "high": 0.20},
    "high":   {"medium": 0.20, "high": 0.60, "urgent": 0.20},
    "urgent": {"high": 0.30, "urgent": 0.70},
}
TICKET_COLUMNS = ["ticket_id", "created_at", "customer_id", "channel", "category", "priority",
                  "subject", "description", "order_id", "status", "resolution",
                  "resolution_time_hours", "related_doc_id"]


def add_typos(text, rng, rate):
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def noisy(text, rng, noise):
    """Make customer text messier: typos, lowercase, 'pls', stray punctuation."""
    if noise <= 0:
        return text
    text = add_typos(text, rng, noise * 0.12)
    if rng.random() < noise * 0.5:
        text = text.lower()
    if rng.random() < noise * 0.3:
        text += rng.choice([" pls help", " urgent!!", " please reply asap", " thanks"])
    if rng.random() < noise * 0.2:
        text = text.replace(".", "..", 1)
    return text


def build_ticket(rng, idx, sc, category_of, customers, end, noise):
    ctx = {
        "order_id": f"ORD-{rng.randint(1000000, 9999999)}",
        "amount": rng.randrange(150, 4500, 10),
        "amount_low": rng.randrange(120, 490, 10),
        "medicine": rng.choice(MEDICINES),
        "device": rng.choice(DEVICES),
        "txn_id": f"TXN{rng.randint(10**9, 10**10 - 1)}",
    }
    w = PRIORITY_WEIGHTS[sc["priority"]]
    priority = rng.choices(list(w), weights=list(w.values()))[0]
    status = rng.choices(STATUSES[0], weights=STATUSES[1])[0]
    created = end - timedelta(days=rng.randint(0, 180), seconds=rng.randint(0, 86399))

    has_resolution = status in ("resolved", "closed")
    return {
        "ticket_id": f"TKT-{idx:05d}",
        "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": rng.choice(customers),
        "channel": rng.choices(CHANNELS[0], weights=CHANNELS[1])[0],
        "category": category_of[sc["doc"]],
        "priority": priority,
        "subject": noisy(rng.choice(sc["subjects"]).format(**ctx), rng, noise * 0.5),
        "description": noisy(rng.choice(sc["bodies"]).format(**ctx), rng, noise),
        "order_id": ctx["order_id"],
        "status": status,
        "resolution": sc["resolution"].format(**ctx) if has_resolution else "",
        "resolution_time_hours": round(rng.lognormvariate(1.6, 0.9), 1) if has_resolution else "",
        "related_doc_id": sc["doc"],
    }


def write_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Generate synthetic policies.csv and tickets.csv for a RAG support copilot.")
    p.add_argument("--n-tickets", type=int, default=1000, help="number of tickets (default 1000)")
    p.add_argument("--noise", type=float, default=0.15, help="customer text noise, 0-1 (default 0.15)")
    p.add_argument("--out-dir", default="data", help="output directory (default: data)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # policies.csv
    policy_rows = [dict(doc_id=d, title=t, content=c, category=cat) for d, t, cat, c in POLICIES]
    write_csv(out / "policies.csv", policy_rows, ["doc_id", "title", "content", "category"])

    # sanity: every scenario points at a real policy, every policy has a scenario
    doc_ids = {r["doc_id"] for r in policy_rows}
    category_of = {r["doc_id"]: r["category"] for r in policy_rows}
    assert {s["doc"] for s in SCENARIOS} <= doc_ids, "scenario references unknown doc_id"
    uncovered = doc_ids - {s["doc"] for s in SCENARIOS}
    assert not uncovered, f"policies without tickets: {sorted(uncovered)}"

    # tickets.csv
    end = datetime(2026, 11, 10, 23, 59, 59)
    customers = [f"CUST-{rng.randint(10000, 99999)}" for _ in range(max(50, a.n_tickets // 3))]
    tickets = [build_ticket(rng, i, rng.choice(SCENARIOS), category_of, customers, end, a.noise)
               for i in range(1, a.n_tickets + 1)]
    tickets.sort(key=lambda r: r["created_at"])
    for i, t in enumerate(tickets, 1):  # IDs follow time order
        t["ticket_id"] = f"TKT-{i:05d}"
    write_csv(out / "tickets.csv", tickets, TICKET_COLUMNS)

    print(f"policies.csv : {len(policy_rows)} documents  -> {out / 'policies.csv'}")
    print(f"tickets.csv  : {len(tickets)} tickets    -> {out / 'tickets.csv'}")
    print("tickets by category:", dict(Counter(t["category"] for t in tickets)))
    print("tickets by status  :", dict(Counter(t["status"] for t in tickets)))


if __name__ == "__main__":
    main()