# CampusWasteSorting — ER Diagram

## 1. ER Diagram (Mermaid)

```mermaid
erDiagram
    users ||--o{ scan_history : identifies
    users ||--o{ system_logs : triggers

    users {
        int user_id PK
        string username UK
        string password_hash
        string role
        string status
    }

    scan_history {
        int id PK
        int user_id FK
        string image_path
        string prediction
        float probability
        string category
        string corrected_category
        string user_feedback
        datetime created_at
    }

    system_logs {
        int id PK
        string action_type
        string operator
        string detail
        datetime created_at
    }

    announcements {
        int id PK
        string content
        int is_active
        datetime created_at
    }

    encyclopedia {
        string category PK
        string content
    }
```

## 2. Table Descriptions

| Table | Description |
|---|---|
| **users** | User auth & role management |
| **scan_history** | Core business records with correction workflow |
| **system_logs** | Audit trail (ban / correction / OTA) |
| **announcements** | System-wide announcements |
| **encyclopedia** | Waste sorting knowledge base CMS |

### scan_history — Correction State Machine

```
NULL (no dispute) → pending (user reported) → resolved (admin corrected)
          ↑                                              │
          └──────────── (user re-disputes) ◄─────────────┘
```
