flowchart TD
    Start[Start] --> MainMenu[Main Menu]
    MainMenu -->|1. Vendor Maintenance| VendorMaintenance[Vendor Maintenance Screen]
    MainMenu -->|2. Invoice Entry| InvoiceEntry[Invoice Entry Screen]
    MainMenu -->|3. Statement Processing| StatementProcessing[Statement Processing Screen]
    MainMenu -->|4. AP Reports| APReports[Accounts Payable Reports]
    MainMenu -->|5. Payment Processing| PaymentProcessing[Payment Processing]
    MainMenu -->|99. Exit| Exit[Exit Program]

    VendorMaintenance --> ActionSelection[Select Action: Add, Change, Delete, Inquire]
    ActionSelection --> EnterVendorDetails[Enter Vendor Details]
    EnterVendorDetails --> ProcessVendorAction[Process Vendor Action]
    ProcessVendorAction --> VendorMaintenance

    InvoiceEntry --> ActionSelectionInvoice[Select Action: Add, Change, Delete, Inquire]
    ActionSelectionInvoice --> EnterInvoiceDetails[Enter Invoice Details]
    EnterInvoiceDetails --> ProcessInvoiceAction[Process Invoice Action]
    ProcessInvoiceAction --> InvoiceEntry

    StatementProcessing --> EnterVendorNumber[Enter Vendor Number]
    EnterVendorNumber --> ProcessAllStatements[Process All Statements: Yes or No]
    ProcessAllStatements --> ProcessStatements[Process Statements]
    ProcessStatements --> StatementProcessing

    APReports --> GenerateReports[Generate Accounts Payable Reports]
    GenerateReports --> APReports

    PaymentProcessing --> EnterPaymentDetails[Enter Payment Details]
    EnterPaymentDetails --> ProcessPayments[Process Payments]
    ProcessPayments --> PaymentProcessing

    Exit --> End[End]
