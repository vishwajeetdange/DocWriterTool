#!/usr/bin/env python3
"""
Test script to verify line-by-line documentation generation functionality.
"""

import os
import sys
from github_doc_generator import GitHubDocGenerator, DocumentationConfig

def test_line_by_line_documentation():
    """Test the line-by-line documentation generation."""
    
    # Sample test contents with different languages
    test_contents = {
        "main.rpg": {
            "type": "file",
            "language": "RPG",
            "size": 1500,
            "content": """H DFTACTGRP(*NO) ACTGRP(*CALLER)
     
     // File specifications
     FCUSTOMER  IF   E           K DISK
     FORDERS    IF   E           K DISK
     FPRINT     O    F  132        PRINTER
     
     // Data structures
     D CustomerDS      DS                  QUALIFIED TEMPLATE
     D  CustNo                       10A
     D  CustName                     30A
     D  Address                      50A
     
     // Standalone fields
     D Counter         S             10I 0
     D TotalAmount     S             15P 2
     
     // Main processing
     C                   READ      CUSTOMER
     C                   DOW       NOT %EOF(CUSTOMER)
     C                   EVAL      Counter = Counter + 1
     C                   CHAIN     CustomerDS.CustNo     ORDERS
     C                   IF        %FOUND(ORDERS)
     C                   EVAL      TotalAmount = TotalAmount + OrderAmount
     C                   ENDIF
     C                   EXCEPT    DetailLine
     C                   READ      CUSTOMER
     C                   ENDDO
     
     C                   SETON                                        LR"""
        },
        "control.clp": {
            "type": "file", 
            "language": "CL Program",
            "size": 800,
            "content": """PGM
     DCL VAR(&LIBRARY) TYPE(*CHAR) LEN(10)
     DCL VAR(&MEMBER) TYPE(*CHAR) LEN(10)
     DCL VAR(&MSGID) TYPE(*CHAR) LEN(7)
     DCL VAR(&MSGDTA) TYPE(*CHAR) LEN(100)
     
     MONMSG MSGID(CPF0000) EXEC(GOTO CMDLBL(ERROR))
     
     CHGVAR VAR(&LIBRARY) VALUE('TESTLIB')
     CHGVAR VAR(&MEMBER) VALUE('TESTMBR')
     
     CRTPF FILE(&LIBRARY/TESTFILE) RCDLEN(80)
     ADDPFM FILE(&LIBRARY/TESTFILE) MBR(&MEMBER)
     
     CPYF FROMFILE(QGPL/SOURCEFILE) TOFILE(&LIBRARY/TESTFILE) +
          FROMMBR(&MEMBER) TOMBR(&MEMBER)
     
     SNDPGMMSG MSG('Processing completed successfully')
     GOTO CMDLBL(END)
     
     ERROR:
     RCVMSG MSGTYPE(*EXCP) MSGDTA(&MSGDTA) MSGID(&MSGID)
     SNDPGMMSG MSG('Error occurred: ' *CAT &MSGID *CAT ' - ' *CAT &MSGDTA)
     
     END:
     ENDPGM"""
        },
        "customer.cbl": {
            "type": "file",
            "language": "COBOL", 
            "size": 2000,
            "content": """IDENTIFICATION DIVISION.
       PROGRAM-ID. CUSTOMER-REPORT.
       AUTHOR. SYSTEM ANALYST.
       
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE ASSIGN TO 'CUSTOMER.DAT'
               ORGANIZATION IS SEQUENTIAL
               ACCESS MODE IS SEQUENTIAL.
           SELECT REPORT-FILE ASSIGN TO 'REPORT.TXT'
               ORGANIZATION IS SEQUENTIAL.
       
       DATA DIVISION.
       FILE SECTION.
       FD  CUSTOMER-FILE.
       01  CUSTOMER-RECORD.
           05  CUST-ID         PIC 9(5).
           05  CUST-NAME       PIC X(30).
           05  CUST-BALANCE    PIC 9(7)V99.
       
       FD  REPORT-FILE.
       01  REPORT-RECORD       PIC X(80).
       
       WORKING-STORAGE SECTION.
       01  WS-EOF              PIC X VALUE 'N'.
       01  WS-TOTAL-BALANCE    PIC 9(9)V99 VALUE ZERO.
       01  WS-RECORD-COUNT     PIC 9(5) VALUE ZERO.
       
       PROCEDURE DIVISION.
       MAIN-PARA.
           OPEN INPUT CUSTOMER-FILE
           OPEN OUTPUT REPORT-FILE
           
           PERFORM READ-CUSTOMER
           PERFORM PROCESS-RECORDS UNTIL WS-EOF = 'Y'
           
           PERFORM WRITE-TOTALS
           
           CLOSE CUSTOMER-FILE
           CLOSE REPORT-FILE
           STOP RUN.
       
       READ-CUSTOMER.
           READ CUSTOMER-FILE
               AT END MOVE 'Y' TO WS-EOF
           END-READ.
       
       PROCESS-RECORDS.
           ADD 1 TO WS-RECORD-COUNT
           ADD CUST-BALANCE TO WS-TOTAL-BALANCE
           PERFORM WRITE-DETAIL
           PERFORM READ-CUSTOMER.
       
       WRITE-DETAIL.
           MOVE CUSTOMER-RECORD TO REPORT-RECORD
           WRITE REPORT-RECORD.
       
       WRITE-TOTALS.
           MOVE SPACES TO REPORT-RECORD
           STRING 'TOTAL RECORDS: ' WS-RECORD-COUNT
               DELIMITED BY SIZE INTO REPORT-RECORD
           WRITE REPORT-RECORD."""
        }
    }
    
    # Create a test configuration with line-by-line docs enabled
    config = DocumentationConfig(
        include_line_by_line_docs=True,
        include_overview=False,
        include_executive_summary=False,
        include_tech_stack=False,
        include_code_structure=False,
        include_loc_analysis=False,
        include_complexity_analysis=False,
        include_features=False,
        include_dependencies=False,
        include_issues=False,
        include_sql_objects=False,
        include_class_diagram=False,
        include_flow_diagram=False,
        include_er_diagram=False,
        include_reference_architecture=False,
        include_loc_chart=False,
        include_complexity_charts=False,
        max_files_to_analyze=3
    )
    
    # Initialize the generator (you'll need to provide your actual Azure OpenAI credentials)
    generator = GitHubDocGenerator(
        deployment_name="your-deployment-name",
        api_key="your-api-key", 
        api_base="your-api-base",
        config=config
    )
    
    print("Testing line-by-line documentation generation...")
    
    # Test file selection
    selected_files = generator.select_key_files(test_contents, max_files=3)
    print(f"Selected {len(selected_files)} files for analysis:")
    for file_path, file_info in selected_files:
        print(f"  - {file_path} ({file_info['language']})")
    
    # Test line-by-line explanation generation
    print("\nGenerating line-by-line explanations...")
    explanations = generator.generate_line_by_line_explanations(test_contents, "test_project")
    
    print(f"Generated explanations for {len(explanations)} files:")
    for file_path, explanation in explanations.items():
        print(f"\n--- {file_path} ---")
        print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_line_by_line_documentation()