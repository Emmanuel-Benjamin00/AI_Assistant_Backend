# US Healthcare Claims and EDI Primer

In the United States, healthcare providers bill insurance payers electronically using ANSI X12 EDI transactions defined under HIPAA. A clearinghouse often sits between the provider and the payer, checking claims for formatting errors and routing them to the correct payer.

The 837 transaction is the claim itself. The 837P (professional) is used for services billed by individual practitioners and many transportation providers, and it replaced the paper CMS-1500 form. The 837I (institutional) is used by hospitals and facilities, and it replaced the paper UB-04 form, which was earlier known as the UB-92.

The 835 transaction is the electronic remittance advice (ERA). The payer sends it back to explain how each claim was paid, including the paid amount, adjustments, and the patient's responsibility. Posting 835 files lets a billing system reconcile payments against the claims that were sent.

The 270 and 271 transactions handle eligibility. A provider sends a 270 inquiry to ask whether a member is covered, and the payer replies with a 271 response describing the member's coverage and benefits.

The 276 and 277 transactions handle claim status. A 276 asks for the status of a submitted claim, and the 277 response reports whether it is received, pending, paid or denied. The 999 acknowledgment confirms whether a submitted EDI file was syntactically accepted.

Adjustments on an 835 are explained with Claim Adjustment Reason Codes (CARC) and Remittance Advice Remark Codes (RARC). Adjustments are grouped as CO (contractual obligation, which the provider writes off), PR (patient responsibility), OA (other adjustment) and PI (payer-initiated reduction).

Procedures and supplies on a claim are identified with HCPCS codes. In non-emergency medical transportation (NEMT), HCPCS codes distinguish the type of trip and vehicle, and mileage is often billed as a separate line using a mileage code with the number of miles as the units.

Medicaid covers non-emergency medical transportation for eligible members. Many state Medicaid programs contract transportation brokers, who take ride requests, assign trips to transportation providers, and handle or route the resulting claims.
