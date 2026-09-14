# HIPAA Privacy Basics for Software Teams

HIPAA is the Health Insurance Portability and Accountability Act, a US law from 1996. Its Privacy Rule controls how protected health information may be used and shared, and its Security Rule sets safeguards for protected health information that is stored or sent electronically.

Protected health information (PHI) is individually identifiable health information held by a covered entity or its business associate. Names, addresses, dates of service, phone numbers, member IDs and medical record numbers become PHI when they are linked to a person's health care or payment for health care.

Covered entities are health plans, health care clearinghouses, and health care providers that transmit health information electronically. A business associate is a vendor that handles PHI on behalf of a covered entity, such as a billing company or a software provider hosting claims data.

Before a business associate receives PHI, the covered entity and the vendor must sign a business associate agreement (BAA). The BAA requires the vendor to protect the data, report breaches, and pass the same obligations to its own subcontractors.

The minimum necessary standard says that a team should use or disclose only the smallest amount of PHI needed for the task. For example, a report about late trips needs trip times and vehicle numbers, not the rider's diagnosis.

Data can be de-identified so that it is no longer PHI. Under the Safe Harbor method, 18 types of identifiers are removed, including names, geographic details smaller than a state, all elements of dates except the year, phone numbers, email addresses and Social Security numbers. The other method is Expert Determination, where a qualified expert certifies that the risk of re-identification is very small.

Before sending PHI to an external AI service, a team should either de-identify the data or confirm that the AI vendor has signed a BAA and that the use is approved. Logs, prompts and error messages must also be checked, because PHI often leaks there by accident.

When unsecured PHI is breached, the covered entity must notify the affected individuals without unreasonable delay and no later than 60 days after discovering the breach. Encrypting data at rest and in transit is a key safeguard, because properly encrypted data that is lost is generally not treated as unsecured PHI.
