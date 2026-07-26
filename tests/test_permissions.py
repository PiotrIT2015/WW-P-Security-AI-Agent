from authentication.auth import (
    AuthenticationManager,
)

from authentication.permissions import (
    AccessDecision,
    DocumentACL,
    PermissionManager,
)


def main():

    auth = AuthenticationManager()

    # =========================================================
    # Users
    # =========================================================

    admin, _ = auth.authenticate(
        "admin",
        "AdminPassword123!",
    ) if auth.get_user("admin") else (
        None,
        None,
    )

    if admin is None:

        auth.create_user(

            "admin",

            "AdminPassword123!",

            roles={
                "admin",
            },

            groups={
                "security",
            },

        )

        admin, _ = auth.authenticate(
            "admin",
            "AdminPassword123!",
        )

    auth.create_user(

        "analyst",

        "AnalystPassword123!",

        roles={
            "analyst",
        },

        groups={
            "security",
        },

    )

    auth.create_user(

        "reader",

        "ReaderPassword123!",

        roles={
            "reader",
        },

        groups={
            "general",
        },

    )

    analyst, _ = auth.authenticate(
        "analyst",
        "AnalystPassword123!",
    )

    reader, _ = auth.authenticate(
        "reader",
        "ReaderPassword123!",
    )

    # =========================================================
    # Permission manager
    # =========================================================

    permissions = PermissionManager()

    # =========================================================
    # Confidential security document
    # =========================================================

    permissions.add_policy(

        DocumentACL(

            document_id="security-001",

            allowed_groups={
                "security",
            },

            allowed_roles={
                "analyst",
            },

            classification="confidential",

            min_role="analyst",

        )
    )

    # =========================================================
    # Public document
    # =========================================================

    permissions.add_policy(

        DocumentACL(

            document_id="public-001",

            classification="public",

        )
    )

    # =========================================================
    # Analyst
    # =========================================================

    result = permissions.authorize(

        analyst,

        "security-001",

    )

    assert (
        result.decision
        == AccessDecision.ALLOW
    )

    print(
        "Analyst confidential access: ALLOW"
    )

    # =========================================================
    # Reader
    # =========================================================

    result = permissions.authorize(

        reader,

        "security-001",

    )

    assert (
        result.decision
        == AccessDecision.DENY
    )

    print(
        "Reader confidential access: DENY"
    )

    # =========================================================
    # Admin
    # =========================================================

    result = permissions.authorize(

        admin,

        "security-001",

    )

    assert (
        result.decision
        == AccessDecision.ALLOW
    )

    print(
        "Admin confidential access: ALLOW"
    )

    # =========================================================
    # Public
    # =========================================================

    result = permissions.authorize(

        reader,

        "public-001",

    )

    assert (
        result.decision
        == AccessDecision.ALLOW
    )

    print(
        "Reader public access: ALLOW"
    )

    # =========================================================
    # Explicit deny
    # =========================================================

    permissions.add_policy(

        DocumentACL(

            document_id="security-002",

            allowed_roles={
                "analyst",
            },

            denied_users={
                analyst.user_id,
            },

            classification="confidential",

        )
    )

    result = permissions.authorize(

        analyst,

        "security-002",

    )

    assert (
        result.decision
        == AccessDecision.DENY
    )

    print(
        "Explicit user DENY: OK"
    )


if __name__ == "__main__":
    main()