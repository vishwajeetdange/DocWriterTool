flowchart TD

    %% Style definitions for all architectural layers
    classDef presentationLayer fill:#d4ffcc,stroke:#3b3,stroke-width:1px
    classDef controllerLayer fill:#ffe0cc,stroke:#f95,stroke-width:1px
    classDef serviceLayer fill:#cce5ff,stroke:#36f,stroke-width:1px
    classDef domainLayer fill:#ffffcc,stroke:#cc0,stroke-width:1px
    classDef dataAccessLayer fill:#e6ccff,stroke:#93f,stroke-width:1px
    classDef infrastructureLayer fill:#e6e6e6,stroke:#666,stroke-width:1px
    classDef databaseLayer fill:#ffd6e0,stroke:#f69,stroke-width:1px
    classDef securityLayer fill:#ffcccc,stroke:#f66,stroke-width:1px

    %% Apply styles to specific component types
    class UI,View,Component,Page,Screen,Form,Template presentationLayer
    class Controller,API,Endpoint,Resource,Route controllerLayer
    class Service,Manager,Handler,Processor,Orchestrator serviceLayer
    class Model,Entity,Domain,DTO,POJO,Bean,VO domainLayer
    class Repository,DAO,Mapper,Store dataAccessLayer
    class Config,Util,Helper,Factory,Provider infrastructureLayer
    class Database,Storage,Cache databaseLayer
    class Security,Auth,Authentication,Authorization securityLayer
            
    title dds_project Reference Architecture

    %% === LAYERS ===
    %% Presentation/UI Layer
    subgraph Presentation [fa:fa-window-maximize]
        UI_Dash[Dashboard fa:fa-window-maximize]
        UI_Monitor[MonitoringUI fa:fa-window-maximize]
    end

    %% API/Controllers Layer
    subgraph Controllers [fa:fa-exchange]
        Ctrl_User[UserAPI fa:fa-exchange]
        Ctrl_Data[DataAPI fa:fa-exchange]
    end

    %% Services Layer
    subgraph Services [fa:fa-cogs]
        Service_Auth[AuthService fa:fa-cogs]
        Service_DataProcessing[DataProcessingService fa:fa-cogs]
    end

    %% Domain Models Layer
    subgraph Domain [fa:fa-cubes]
        Domain_User[UserModel fa:fa-cubes]
        Domain_Data[DataModel fa:fa-cubes]
    end

    %% Data Access Layer
    subgraph DAO [fa:fa-database]
        DAO_User[UserDAO fa:fa-database]
        DAO_Data[DataDAO fa:fa-database]
    end

    %% Database Layer
    subgraph Database [fa:fa-server]
        DB_Main[MainDatabase fa:fa-server]
        DB_Logs[LogDatabase fa:fa-server]
    end

    %% Infrastructure Layer
    subgraph Infrastructure [fa:fa-cloud]
        Infra_Security[SecurityConfig fa:fa-shield]
        Infra_Config[ConfigService fa:fa-wrench]
        Infra_Cloud[CloudService fa:fa-cloud]
    end

    %% === CONNECTIONS ===
    %% UI to Controllers
    UI_Dash --> Ctrl_User
    UI_Monitor --> Ctrl_Data

    %% Controllers to Services
    Ctrl_User --> Service_Auth
    Ctrl_Data --> Service_DataProcessing

    %% Services to Domain and DAO
    Service_Auth --> Domain_User
    Service_DataProcessing --> Domain_Data
    Service_Auth --> DAO_User
    Service_DataProcessing --> DAO_Data

    %% DAO to Database
    DAO_User --> DB_Main
    DAO_Data --> DB_Logs

    %% Infrastructure Cross-Cutting Concerns
    Infra_Security -.-> Ctrl_User
    Infra_Security -.-> Service_Auth
    Infra_Config -.-> Service_DataProcessing
    Infra_Cloud -.-> DB_Main

    %% === STYLING ===
    classDef presentation fill:#d4ffcc,stroke:#000,stroke-width:1px;
    classDef controller fill:#ffe0cc,stroke:#000,stroke-width:1px;
    classDef service fill:#cce5ff,stroke:#000,stroke-width:1px;
    classDef domain fill:#ffffcc,stroke:#000,stroke-width:1px;
    classDef dao fill:#e6ccff,stroke:#000,stroke-width:1px;
    classDef database fill:#ffd6e0,stroke:#000,stroke-width:1px;
    classDef infra fill:#e6e6e6,stroke:#000,stroke-width:1px;

    class UI_Dash,UI_Monitor presentation
    class Ctrl_User,Ctrl_Data controller
    class Service_Auth,Service_DataProcessing service
    class Domain_User,Domain_Data domain
    class DAO_User,DAO_Data dao
    class DB_Main,DB_Logs database
    class Infra_Security,Infra_Config,Infra_Cloud infra
