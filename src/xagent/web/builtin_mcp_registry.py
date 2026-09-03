from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

OAUTH_PROVIDERS_TABLE = sa.table(
    "oauth_providers",
    sa.column("provider_name", sa.String),
    sa.column("name", sa.String),
    sa.column("client_id", sa.String),
    sa.column("client_secret", sa.String),
    sa.column("auth_url", sa.String),
    sa.column("token_url", sa.String),
    sa.column("redirect_uri", sa.String),
    sa.column("userinfo_url", sa.String),
    sa.column("user_id_path", sa.String),
    sa.column("email_path", sa.String),
    sa.column("default_scopes", sa.JSON),
)

PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("icon", sa.String),
    sa.column("transport", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("category", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("is_visible_in_connector", sa.Boolean),
    sa.column("launch_config", sa.JSON),
)


def get_builtin_oauth_provider_rows() -> list[dict[str, Any]]:
    return [
        {
            "provider_name": "google",
            "name": "Google",
            "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
            "auth_url": "https://accounts.google.com/o/oauth2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "redirect_uri": os.environ.get("GOOGLE_REDIRECT_URI", ""),
            "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
            "user_id_path": "id",
            "email_path": "email",
            "default_scopes": [
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            ],
        },
        {
            "provider_name": "linkedin",
            "name": "LinkedIn",
            "client_id": os.environ.get("LINKEDIN_CLIENT_ID", ""),
            "client_secret": os.environ.get("LINKEDIN_CLIENT_SECRET", ""),
            "auth_url": "https://www.linkedin.com/oauth/v2/authorization",
            "token_url": "https://www.linkedin.com/oauth/v2/accessToken",
            "redirect_uri": os.environ.get("LINKEDIN_REDIRECT_URI", ""),
            "userinfo_url": "https://api.linkedin.com/v2/userinfo",
            "user_id_path": "sub",
            "email_path": "email",
            "default_scopes": ["openid", "profile", "email", "w_member_social"],
        },
        {
            "provider_name": "microsoft",
            "name": "Microsoft",
            "client_id": os.environ.get("MICROSOFT_CLIENT_ID", ""),
            "client_secret": os.environ.get("MICROSOFT_CLIENT_SECRET", ""),
            "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
            "redirect_uri": os.environ.get("MICROSOFT_REDIRECT_URI", ""),
            "userinfo_url": "https://graph.microsoft.com/v1.0/me",
            "user_id_path": "id",
            "email_path": "userPrincipalName",
            "default_scopes": ["User.Read"],
        },
        {
            "provider_name": "hubspot",
            "name": "HubSpot",
            "client_id": os.environ.get("HUBSPOT_CLIENT_ID", ""),
            "client_secret": os.environ.get("HUBSPOT_CLIENT_SECRET", ""),
            "auth_url": "https://app.hubspot.com/oauth/authorize",
            "token_url": "https://api.hubapi.com/oauth/v1/token",
            "redirect_uri": os.environ.get("HUBSPOT_REDIRECT_URI", ""),
            # HubSpot's token-info endpoint takes the token in the URL path
            # rather than a Bearer header; the callback substitutes the
            # {{access_token}} placeholder before issuing the GET.
            "userinfo_url": "https://api.hubapi.com/oauth/v1/access-tokens/{{access_token}}",
            "user_id_path": "user_id",
            "email_path": "user",
            "default_scopes": ["oauth"],
        },
        {
            "provider_name": "meta",
            "name": "Meta",
            "client_id": os.environ.get("META_CLIENT_ID", ""),
            "client_secret": os.environ.get("META_CLIENT_SECRET", ""),
            "auth_url": "https://www.facebook.com/v25.0/dialog/oauth",
            "token_url": "https://graph.facebook.com/v25.0/oauth/access_token",
            "redirect_uri": os.environ.get("META_REDIRECT_URI", ""),
            "userinfo_url": "https://graph.facebook.com/v25.0/me?fields=id,email",
            "user_id_path": "id",
            "email_path": "email",
            "default_scopes": ["public_profile"],
        },
        {
            "provider_name": "slack",
            "name": "Slack",
            "client_id": os.environ.get("SLACK_CLIENT_ID", ""),
            "client_secret": os.environ.get("SLACK_CLIENT_SECRET", ""),
            "auth_url": "https://slack.com/oauth/v2/authorize",
            "token_url": "https://slack.com/api/oauth.v2.access",
            "redirect_uri": os.environ.get("SLACK_REDIRECT_URI", ""),
            "userinfo_url": "https://slack.com/api/auth.test",
            # auth.test never returns an email for a bot token; the workspace
            # name ("team") is deliberately stored in the email slot because
            # UserOAuth.email is only consumed as the "connected account"
            # display label for non-gmail providers — without it the Slack
            # connection would show up unlabeled in the connector UI.
            "user_id_path": "team_id",
            "email_path": "team",
            "default_scopes": [
                "chat:write",
                "chat:write.public",
                "channels:read",
                "channels:history",
                "channels:join",
                "groups:read",
                "groups:history",
                "im:read",
                "im:history",
                "mpim:read",
                "mpim:history",
                "reactions:write",
                "files:write",
            ],
        },
        {
            "provider_name": "zoom",
            "name": "Zoom",
            "client_id": os.environ.get("ZOOM_CLIENT_ID", ""),
            "client_secret": os.environ.get("ZOOM_CLIENT_SECRET", ""),
            "auth_url": "https://zoom.us/oauth/authorize",
            "token_url": "https://zoom.us/oauth/token",
            "redirect_uri": os.environ.get("ZOOM_REDIRECT_URI", ""),
            "userinfo_url": "https://api.zoom.us/v2/users/me",
            "user_id_path": "id",
            "email_path": "email",
            # Identity-only for this connector, similar in spirit to google
            # (userinfo.email/profile) and meta (public_profile) — though
            # it isn't a strict rule across every provider here (linkedin's
            # default_scopes includes the functional w_member_social write
            # scope). The functional scopes this connector actually calls
            # live on the app row's oauth_scopes below and are merged in at
            # authorize time by _merge_oauth_scopes, so listing them here
            # too would just be a second place scope changes have to be
            # kept in sync.
            "default_scopes": ["user:read:user"],
        },
        {
            "provider_name": "github",
            "name": "GitHub",
            "client_id": os.environ.get("GITHUB_CLIENT_ID", ""),
            "client_secret": os.environ.get("GITHUB_CLIENT_SECRET", ""),
            "auth_url": "https://github.com/login/oauth/authorize",
            "token_url": "https://github.com/login/oauth/access_token",
            "redirect_uri": os.environ.get("GITHUB_REDIRECT_URI", ""),
            "userinfo_url": "https://api.github.com/user",
            "user_id_path": "id",
            # GitHub's /user "email" field is null unless the account has a
            # public email set or the connecting token carries user:email
            # (only requested via this app's own oauth_scopes below, not the
            # provider's bare default_scopes) -- "login" (the username) is
            # always present on any valid token and is what the connector UI
            # actually needs for a display label, same rationale as slack's
            # team-name-in-email-slot above.
            "email_path": "login",
            # Identity-only for this provider row, same pattern as zoom
            # above: the functional scopes (repo access, email) live on
            # the app row's oauth_scopes and are merged in at authorize
            # time by _merge_oauth_scopes.
            "default_scopes": ["read:user"],
        },
        {
            "provider_name": "intercom",
            "name": "Intercom",
            "client_id": os.environ.get("INTERCOM_CLIENT_ID", ""),
            "client_secret": os.environ.get("INTERCOM_CLIENT_SECRET", ""),
            # The single "app.intercom.com" authorize host is claimed to serve
            # every region: per Intercom's community answer to this exact
            # question (community.intercom.com/api-webhooks-23/do-we-need-to
            # -create-multiple-oauth-apps-per-data-region-2522), a public app
            # is replicated across US/EU/AU automatically once it clears
            # Intercom App Review, so there is no per-region app or authorize
            # URL to manage. This is a community answer, not primary API
            # reference docs, and has not been verified end-to-end against a
            # live non-US workspace -- if it turns out to be wrong, EU/AU
            # connects would need their own provider row instead of sharing
            # this one.
            "auth_url": "https://app.intercom.com/oauth",
            # "api.intercom.io" (no region prefix) auto-routes each request to
            # the workspace's actual hosting region (US/EU/AU) based on the
            # token/workspace being acted on -- this part IS documented in
            # Intercom's primary REST API reference (developers.intercom.com/
            # docs/build-an-integration/learn-more/rest-apis, "About our REST
            # API"). Combined with the authorize-host claim above, this is
            # what lets one provider row, unlike the hosted MCP server's
            # separate mcp.intercom.com / mcp.eu.intercom.com endpoints (which
            # don't support AU workspaces at all), cover all three regions.
            "token_url": "https://api.intercom.io/auth/eagle/token",
            "redirect_uri": os.environ.get("INTERCOM_REDIRECT_URI", ""),
            "userinfo_url": "https://api.intercom.io/me",
            "user_id_path": "id",
            "email_path": "email",
            # Intercom has no `scope` authorize-URL param at all -- granted
            # permissions come entirely from the app's Authentication
            # settings in the Developer Hub, the same out-of-band model as
            # meta's config_id above. Leaving this empty means
            # generic_oauth_login's `elif scope_str:` guard (api/auth.py)
            # never adds a scope param for this provider, which matches
            # Intercom's actual behavior instead of sending a param it
            # ignores.
            "default_scopes": [],
        },
        {
            "provider_name": "salesforce",
            "name": "Salesforce",
            "client_id": os.environ.get("SALESFORCE_CLIENT_ID", ""),
            "client_secret": os.environ.get("SALESFORCE_CLIENT_SECRET", ""),
            # The generic login.salesforce.com/test.salesforce.com hosts
            # (rather than a customer's own My Domain, e.g.
            # acme.my.salesforce.com) work for any org: Salesforce resolves
            # the actual org during login and issues a token scoped to it,
            # returning that org's real API host as `instance_url` in the
            # token response -- see the userinfo_url note below and
            # UserOAuth.instance_url (api/auth.py persists it generically
            # for any provider that returns this key).
            "auth_url": "https://login.salesforce.com/services/oauth2/authorize",
            "token_url": "https://login.salesforce.com/services/oauth2/token",
            "redirect_uri": os.environ.get("SALESFORCE_REDIRECT_URI", ""),
            # Salesforce's OIDC userinfo endpoint IS a fixed host
            # (login.salesforce.com, same as auth_url/token_url -- see
            # salesforce.py's USERINFO_URL), not the per-org instance_url
            # used by every other endpoint. Left empty anyway, on purpose:
            # populating it here would add a network round-trip to every
            # OAuth connect just to fill in email/provider_user_id, which
            # this connector doesn't otherwise need -- identity is instead
            # fetched lazily, on demand, via salesforce_get_current_user.
            # UserOAuth.email/provider_user_id stay NULL for every Salesforce
            # grant as a result; is_connected still works from access_token
            # alone (see test_salesforce_oauth.py's dedicated coverage).
            "userinfo_url": "",
            "user_id_path": "user_id",
            "email_path": "email",
            # refresh_token: required to get a refresh_token back at all.
            # openid: required for the OIDC userinfo endpoint
            # salesforce_get_current_user calls.
            "default_scopes": ["api", "refresh_token", "openid"],
        },
        {
            "provider_name": "deputy",
            "name": "Deputy",
            "client_id": os.environ.get("DEPUTY_CLIENT_ID", ""),
            "client_secret": os.environ.get("DEPUTY_CLIENT_SECRET", ""),
            "auth_url": "https://once.deputy.com/my/oauth/login",
            "token_url": "https://once.deputy.com/my/oauth/access_token",
            "redirect_uri": os.environ.get("DEPUTY_REDIRECT_URI", ""),
            # Deputy has no fixed userinfo host -- each customer's actual API
            # host (e.g. "acme.au.deputy.com", no scheme) is returned as
            # `endpoint` in the token response instead, similar in spirit to
            # Salesforce's instance_url above but under a different response
            # key and without a scheme, so api/auth.py normalizes it into
            # UserOAuth.instance_url itself rather than reusing the generic
            # token_data.get("instance_url") persistence every other provider
            # shares. Left empty so generic_oauth_callback's `elif
            # userinfo_url and access_token:` branch is skipped; identity
            # instead comes from a dedicated `elif provider.lower() ==
            # "deputy"` branch there that calls the per-install
            # GET /api/v1/me once instance_url is known.
            "userinfo_url": "",
            "user_id_path": "",
            "email_path": "",
            # Deputy's only documented OAuth scope. Not a permission scope in
            # the usual sense -- it just tells Deputy to also issue a
            # refresh_token -- but the authorize, code-exchange, and refresh
            # requests all require it to be present verbatim.
            "default_scopes": ["longlife_refresh_token"],
        },
        {
            "provider_name": "linear",
            "name": "Linear",
            "client_id": os.environ.get("LINEAR_CLIENT_ID", ""),
            "client_secret": os.environ.get("LINEAR_CLIENT_SECRET", ""),
            "auth_url": "https://linear.app/oauth/authorize",
            "token_url": "https://api.linear.app/oauth/token",
            "redirect_uri": os.environ.get("LINEAR_REDIRECT_URI", ""),
            # Linear's API is GraphQL-only (POST-only, nested response shape)
            # -- there is no flat REST endpoint this callback's GET-plus-flat-
            # dict.get() userinfo lookup can use. Left empty so that lookup is
            # skipped entirely (generic_oauth_callback's `if userinfo_url and
            # access_token:` guard); identity comes instead from a dedicated
            # `elif provider.lower() == "linear"` branch there that runs a
            # `viewer` GraphQL query (_fetch_linear_viewer_identity), which
            # also doubles as post-exchange token verification.
            "userinfo_url": "",
            "user_id_path": "id",
            "email_path": "email",
            # Identity-only convention (see zoom's comment above) — the
            # functional "write" scope this connector actually calls for
            # lives on the app row's oauth_scopes below and is merged in at
            # authorize time by _merge_oauth_scopes, so listing it here too
            # would just be a second place scope changes have to be kept in
            # sync, and would survive a narrowing of the app row's scopes
            # since _merge_oauth_scopes is a pure union.
            "default_scopes": ["read"],
        },
        {
            "provider_name": "jira",
            "name": "Jira",
            "client_id": os.environ.get("JIRA_CLIENT_ID", ""),
            "client_secret": os.environ.get("JIRA_CLIENT_SECRET", ""),
            "auth_url": "https://auth.atlassian.com/authorize",
            "token_url": "https://auth.atlassian.com/oauth/token",
            "redirect_uri": os.environ.get("JIRA_REDIRECT_URI", ""),
            "userinfo_url": "https://api.atlassian.com/me",
            "user_id_path": "account_id",
            "email_path": "email",
            # offline_access is required to get a refresh_token back at all
            # (Atlassian omits it from the token response otherwise); read/
            # write:jira-work cover issue and comment access, read:jira-user
            # covers the user-search tool, and read:me backs jira_get_current_user.
            "default_scopes": [
                "offline_access",
                "read:jira-work",
                "write:jira-work",
                "read:jira-user",
                "read:me",
            ],
        },
        {
            "provider_name": "employment-hero",
            "name": "Employment Hero",
            "client_id": os.environ.get("EMPLOYMENT_HERO_CLIENT_ID", ""),
            "client_secret": os.environ.get("EMPLOYMENT_HERO_CLIENT_SECRET", ""),
            "auth_url": "https://oauth.employmenthero.com/oauth2/authorize",
            "token_url": "https://oauth.employmenthero.com/oauth2/token",
            "redirect_uri": os.environ.get("EMPLOYMENT_HERO_REDIRECT_URI", ""),
            # Employment Hero has no OIDC-style "me"/identity endpoint — its
            # closest resource, GET /api/v1/organisations, returns a *list*
            # of organisations the token can access rather than a single
            # object with an id/email shape this callback's flat
            # user_id_path/email_path lookup could use. Left empty, same
            # tradeoff as salesforce/linear's rows above: generic_oauth_
            # callback's `if userinfo_url and access_token:` guard skips the
            # lookup, so UserOAuth.email/provider_user_id stay NULL for
            # every grant — is_connected still works from access_token
            # alone, and an agent calls employment_hero_list_organisations
            # to discover the organisation_id every other tool needs anyway.
            "userinfo_url": "",
            "user_id_path": "id",
            "email_path": "email",
            # Employment Hero has no separate identity-only scope (unlike
            # google/zoom/github above) — every scope here is a functional
            # read permission this connector's tools actually call, so they
            # live entirely on the app row's oauth_scopes below and are
            # merged in at authorize time by _merge_oauth_scopes, same
            # no-default-scopes convention as intercom's row above.
            "default_scopes": [],
        },
    ]


def get_builtin_public_mcp_app_rows() -> list[dict[str, Any]]:
    return [
        {
            "app_id": "linkedin",
            "name": "LinkedIn",
            "description": "Access LinkedIn to retrieve basic user profiles and manage your created posts.",
            "icon": "https://www.google.com/s2/favicons?domain=linkedin.com&sz=128",
            "transport": "oauth",
            "provider_name": "linkedin",
            "category": "CRM",
            "oauth_scopes": ["openid", "profile", "email", "w_member_social"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.linkedin"],
                "env_mapping": {"LINKEDIN_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "gmail",
            "name": "Gmail",
            "description": "Connect to your Gmail inbox to read, search, draft, and send emails.",
            "icon": "https://www.google.com/s2/favicons?domain=mail.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Communication",
            "oauth_scopes": ["https://www.googleapis.com/auth/gmail.modify"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.gmail"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-drive",
            "name": "Google Drive",
            "description": "Access Google Drive to search for files, read documents, and manage your cloud storage.",
            "icon": "https://www.google.com/s2/favicons?domain=drive.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Support",
            "oauth_scopes": ["https://www.googleapis.com/auth/drive"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_drive"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-calendar",
            "name": "Google Calendar",
            "description": "Connect to Google Calendar to manage events, schedule meetings, and view your daily agenda.",
            "icon": "https://www.google.com/s2/favicons?domain=calendar.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Scheduling",
            "oauth_scopes": ["https://www.googleapis.com/auth/calendar.events"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.calendar"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-docs",
            "name": "Google Docs",
            "description": "Connect to Google Docs to create documents from Markdown or text, read document content, and edit text.",
            "icon": "https://www.google.com/s2/favicons?domain=docs.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Productivity",
            "oauth_scopes": [
                "https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/drive.file",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_docs"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-slides",
            "name": "Google Slides",
            "description": "Connect to Google Slides to create presentations, add slides, and read slide content.",
            "icon": "https://www.google.com/s2/favicons?domain=slides.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Productivity",
            "oauth_scopes": [
                "https://www.googleapis.com/auth/presentations",
                "https://www.googleapis.com/auth/drive.file",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_slides"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-sheets",
            "name": "Google Sheets",
            "description": "Connect to Google Sheets to read and update ranges, append rows, manage sheets, and create spreadsheets.",
            "icon": "https://www.google.com/s2/favicons?domain=sheets.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Productivity",
            "oauth_scopes": [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_sheets"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-ads",
            "name": "Google Ads",
            "description": "Connect to Google Ads to list accessible accounts, inspect campaigns, and run GAQL reports.",
            "icon": "https://www.google.com/s2/favicons?domain=ads.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Marketing",
            "oauth_scopes": ["https://www.googleapis.com/auth/adwords"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_ads"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
                "static_env": {
                    "GOOGLE_ADS_DEVELOPER_TOKEN": "GOOGLE_ADS_DEVELOPER_TOKEN"
                },
            },
        },
        {
            "app_id": "google-analytics",
            "name": "Google Analytics",
            "description": "Connect to Google Analytics (GA4) to list properties and run performance reports.",
            "icon": "https://www.google.com/s2/favicons?domain=analytics.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Marketing",
            "oauth_scopes": ["https://www.googleapis.com/auth/analytics.readonly"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_analytics"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-search-console",
            "name": "Google Search Console",
            "description": "Connect to Google Search Console to list verified sites, query search analytics (clicks, impressions, CTR, position), inspect URLs, and list sitemaps.",
            "icon": "https://www.google.com/s2/favicons?domain=search.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            # "Analytics" (not "Marketing", which google-analytics uses —
            # a pre-existing mismatch, not a precedent to repeat) since this
            # connector is squarely a search-analytics tool.
            "category": "Analytics",
            "oauth_scopes": ["https://www.googleapis.com/auth/webmasters.readonly"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_search_console"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "hubspot",
            "name": "HubSpot",
            "description": "Connect to HubSpot CRM and Marketing Hub to search, create, and update contacts and companies, read deals, log notes, read forms and submissions, pull traffic analytics reports, and read marketing emails and campaigns.",
            "icon": "https://www.google.com/s2/favicons?domain=hubspot.com&sz=128",
            "transport": "oauth",
            "provider_name": "hubspot",
            "category": "CRM",
            "oauth_scopes": [
                "crm.objects.contacts.read",
                "crm.objects.contacts.write",
                "crm.objects.companies.read",
                "crm.objects.companies.write",
                "crm.objects.deals.read",
                "forms",
            ],
            # All three are tier-gated, each confirmed against HubSpot's own
            # docs: business-intelligence requires Marketing Hub Basic+ (the
            # traffic analytics tool isn't available on Free/Starter);
            # marketing-email requires Enterprise or the transactional email
            # add-on; marketing.campaigns.read requires Professional+ (the
            # Campaigns API docs state this explicitly). Requesting any of
            # them as required scopes would block the whole OAuth
            # authorization (and therefore every CRM tool too) for portals
            # below those tiers. Separately: GET /marketing/v3/emails (used
            # by hubspot_list_marketing_emails and
            # hubspot_get_marketing_email_statistics) accepts marketing-email
            # OR content OR transactional-email - marketing-email is used
            # here as the most narrowly-scoped of the three that still
            # covers both read calls; content is broader (also grants
            # pages/blog/campaigns access this connector doesn't use) and
            # transactional-email's applicability to non-transactional
            # marketing emails is unconfirmed.
            # Sent via the authorize request's optional_scope parameter
            # instead (see api/auth.py) so those portals still connect
            # successfully and only hubspot_get_analytics_report /
            # hubspot_list_marketing_emails / hubspot_get_marketing_email_statistics /
            # hubspot_list_campaigns / hubspot_get_campaign_metrics fail at
            # call time if the tier doesn't grant them. This pairing must
            # also be reflected in this app's scope configuration in the
            # HubSpot Developer Dashboard - HubSpot blocks installation if a
            # scope's required/optional designation there disagrees with
            # which parameter it arrives in.
            "optional_oauth_scopes": [
                "business-intelligence",
                "marketing-email",
                "marketing.campaigns.read",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.hubspot"],
                "env_mapping": {"HUBSPOT_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "teams",
            "name": "Teams",
            "description": "Connect to Microsoft Teams to list teams, read channels and chats, and send messages.",
            "icon": "https://www.google.com/s2/favicons?domain=teams.microsoft.com&sz=128",
            "transport": "oauth",
            "provider_name": "microsoft",
            "category": "Communication",
            "oauth_scopes": [
                "Team.ReadBasic.All",
                "Channel.ReadBasic.All",
                "TeamMember.Read.All",
                "ChannelMessage.Read.All",
                "ChannelMessage.Send",
                "Chat.ReadWrite",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.teams"],
                "env_mapping": {"AUTH_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "outlook",
            "name": "Outlook",
            "description": "Connect to Outlook to work with mail, calendar events, contacts, and profile data.",
            "icon": "https://www.google.com/s2/favicons?domain=outlook.office.com&sz=128",
            "transport": "oauth",
            "provider_name": "microsoft",
            "category": "Communication",
            "oauth_scopes": [
                "Mail.Read",
                "Mail.Send",
                "Calendars.ReadWrite",
                "Contacts.Read",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.outlook"],
                "env_mapping": {"AUTH_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "onedrive",
            "name": "OneDrive",
            "description": "Connect to OneDrive to browse files, download content, and manage cloud storage.",
            "icon": "https://www.google.com/s2/favicons?domain=onedrive.live.com&sz=128",
            "transport": "oauth",
            "provider_name": "microsoft",
            "category": "Storage",
            "oauth_scopes": ["Files.ReadWrite"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.onedrive"],
                "env_mapping": {"AUTH_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "facebook",
            "name": "Facebook Pages",
            "description": "Connect to Facebook Pages to discover managed pages and publish page posts.",
            "icon": "https://www.google.com/s2/favicons?domain=facebook.com&sz=128",
            "transport": "oauth",
            "provider_name": "meta",
            "category": "Marketing",
            "oauth_scopes": [
                "pages_show_list",
                "pages_read_engagement",
                "pages_manage_posts",
                "pages_read_user_content",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.facebook"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "instagram",
            "name": "Instagram",
            "description": "Connect to Instagram professional accounts through Meta Graph API.",
            "icon": "https://www.google.com/s2/favicons?domain=instagram.com&sz=128",
            "transport": "oauth",
            "provider_name": "meta",
            "category": "Marketing",
            "oauth_scopes": [
                "pages_show_list",
                "pages_read_engagement",
                "instagram_basic",
                "instagram_content_publish",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.instagram"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "zoom",
            "name": "Zoom",
            "description": "Connect to Zoom to look up meetings, and read cloud recordings and transcripts.",
            "icon": "https://www.google.com/s2/favicons?domain=zoom.us&sz=128",
            "transport": "oauth",
            "provider_name": "zoom",
            "category": "Scheduling",
            "oauth_scopes": [
                "meeting:read:meeting",
                "meeting:read:list_meetings",
                "meeting:read:past_meeting",
                "cloud_recording:read:list_recording_files",
                "cloud_recording:read:meeting_transcript",
                "user:read:user",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.zoom"],
                "env_mapping": {"ZOOM_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-maps",
            "name": "Google Maps",
            "description": "Geocoding, directions, place search, and more via the Google Maps APIs.",
            "icon": "https://www.google.com/s2/favicons?domain=maps.google.com&sz=128",
            "transport": "stdio",
            "provider_name": None,
            "category": "Productivity",
            "oauth_scopes": None,
            "is_visible_in_connector": True,
            # Key-based (non-oauth): connected via POST /api/mcp/apps/{id}/connect.
            # required_env tells the connector which secret(s) to prompt for.
            "launch_config": {
                "command": "npx",
                "args": ["-y", "@cablate/mcp-google-map", "--stdio"],
                "required_env": ["GOOGLE_MAPS_API_KEY"],
            },
        },
        {
            "app_id": "slack",
            "name": "Slack",
            "description": "Connect to Slack to search and read channel, thread, and DM history, post messages and replies, react to messages, upload files, and join public channels when asked to, e.g. incident summaries and recommended fixes.",
            "icon": "https://www.google.com/s2/favicons?domain=slack.com&sz=128",
            "transport": "oauth",
            "provider_name": "slack",
            "category": "Communication",
            "oauth_scopes": [
                "chat:write",
                "chat:write.public",
                "channels:read",
                "channels:history",
                "channels:join",
                "groups:read",
                "groups:history",
                "im:read",
                "im:history",
                "mpim:read",
                "mpim:history",
                "reactions:write",
                "files:write",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.slack"],
                "env_mapping": {"SLACK_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "granola",
            "name": "Granola",
            "description": "Connect to Granola to search and read your meeting notes, transcripts and action items through Granola's hosted MCP server.",
            "icon": "https://www.google.com/s2/favicons?domain=granola.ai&sz=128",
            "transport": "streamable_http",
            "provider_name": None,
            "category": "Productivity",
            "oauth_scopes": None,
            "is_visible_in_connector": True,
            # Remote MCP (mcp_oauth): Granola hosts the MCP server itself and
            # exposes its own tools — there is no local module to launch. Users
            # connect via POST /api/mcp/apps/{id}/oauth/connect (per-user OAuth
            # Authorization Code + PKCE); Granola issues no static client
            # credentials, so the flow relies on Dynamic Client Registration.
            "launch_config": {
                "url": "https://mcp.granola.ai/mcp",
                "auth": {"type": "mcp_oauth"},
            },
        },
        {
            "app_id": "notion",
            "name": "Notion",
            "description": "Connect to Notion to search your workspace and read, create and update pages and databases through Notion's hosted MCP server.",
            "icon": "https://www.google.com/s2/favicons?domain=notion.so&sz=128",
            "transport": "streamable_http",
            "provider_name": None,
            "category": "Productivity",
            "oauth_scopes": None,
            "is_visible_in_connector": True,
            # Remote MCP (mcp_oauth), same shape as Granola: Notion hosts the
            # MCP server itself and exposes its own tools — there is no local
            # module to launch. Users connect via POST
            # /api/mcp/apps/{id}/oauth/connect (per-user OAuth Authorization
            # Code + PKCE); Notion supports Dynamic Client Registration, so no
            # static client credentials are required.
            "launch_config": {
                "url": "https://mcp.notion.com/mcp",
                "auth": {"type": "mcp_oauth"},
            },
        },
        {
            "app_id": "aws",
            "name": "AWS",
            "description": "Connect to AWS to check CloudWatch alarms/metrics/logs, DynamoDB health, and SQS queue depth.",
            "icon": "https://www.google.com/s2/favicons?domain=aws.amazon.com&sz=128",
            "transport": "stdio",
            "provider_name": None,
            "category": "Operations",
            "oauth_scopes": None,
            "is_visible_in_connector": True,
            # Key-based (non-oauth), like google-maps: the user supplies an
            # access key at connect time. Cross-account access goes through the
            # per-call role_arn tool parameter (STS AssumeRole inside the MCP
            # module), not a fourth env key — required_env has no optional slots.
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.aws"],
                "required_env": [
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_REGION",
                ],
            },
        },
        {
            "app_id": "posthog",
            "name": "PostHog",
            "description": "Connect to PostHog Cloud (US or EU) to query events and persons via HogQL, and read insights, feature flags, dashboards, and annotations. Self-hosted PostHog is not supported.",
            "icon": "https://www.google.com/s2/favicons?domain=posthog.com&sz=128",
            "transport": "stdio",
            "provider_name": None,
            # "Analytics" is already one of the connect dialog's fixed
            # sidebar category filters (connect-mcp-dialog.tsx) with no
            # connector using it before this one -- an always-empty filter
            # button. google-analytics uses "Marketing" instead, but that's
            # its own pre-existing mismatch, not a reason to leave this
            # button empty too.
            "category": "Analytics",
            "oauth_scopes": None,
            "is_visible_in_connector": True,
            # Key-based (non-oauth), like aws/google-maps: PostHog's only
            # documented OAuth path is Client ID Metadata Document (CIMD),
            # where the client_id is a URL hosting a metadata document *we*
            # would have to serve publicly -- infrastructure this connector
            # doesn't have, and a materially different model (no client
            # secret at all) than every oauth_providers row in this file.
            # Personal API keys sidestep that entirely: each user generates
            # their own (posthog.com/docs/api/personal-api-keys), same
            # no-review self-serve bar as an OAuth App, without needing
            # xagent to host anything. POSTHOG_HOST is required alongside
            # the key because US/EU Cloud are separate deployments -- a key
            # from one host is not valid against the other (unlike, say,
            # Intercom's single auto-routing API host). posthog.py's
            # _base_url() restricts this to exactly those two hosts (not a
            # broader "*.posthog.com" allowlist): self-hosted PostHog is
            # not a documented target of this connector, so there's no
            # reason to let this key be sent anywhere else.
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.posthog"],
                "required_env": ["POSTHOG_API_KEY", "POSTHOG_HOST"],
            },
        },
        {
            "app_id": "mixpanel",
            "name": "Mixpanel",
            "description": "Connect to Mixpanel with a Service Account to query events, segmentation, retention, funnels, and user profiles, and export raw event data. Supports US, EU, and India data residency.",
            "icon": "https://www.google.com/s2/favicons?domain=mixpanel.com&sz=128",
            "transport": "stdio",
            "provider_name": None,
            "category": "Analytics",
            "oauth_scopes": None,
            "is_visible_in_connector": True,
            # Key-based (non-oauth), like posthog/stripe: Mixpanel's Query/
            # Export API authenticates with a Service Account's username +
            # secret over HTTP Basic, generated self-serve by the project
            # owner (no partner application or review). MIXPANEL_PROJECT_ID
            # is required because the Query API is project-scoped, not
            # account-scoped -- every call needs it regardless of endpoint.
            # MIXPANEL_REGION is in required_env (there is no optional_env
            # concept anywhere in the connect flow, per api/mcp.py) even
            # though mixpanel.py itself treats a missing/empty value as
            # "us" -- omitting it here would leave EU/India users with no
            # way to select their region through the connect UI at all.
            # Unlike POSTHOG_HOST, this only ever selects one of Mixpanel's
            # three fixed data-residency host pairs (us/eu/in, see
            # mixpanel.py's _REGION_HOSTS), not an arbitrary host, so there
            # is no user-supplied-URL SSRF surface to validate here.
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.mixpanel"],
                "required_env": [
                    "MIXPANEL_SERVICE_ACCOUNT_USERNAME",
                    "MIXPANEL_SERVICE_ACCOUNT_SECRET",
                    "MIXPANEL_PROJECT_ID",
                    "MIXPANEL_REGION",
                ],
            },
        },
        {
            # "chrome-devtools", not the generic "chrome", matching the
            # upstream package name and the vendor-scoped ids every other
            # seeded app uses. Known, accepted caveat: api/mcp.py couples
            # catalog apps to user-server names by normalized id AND display
            # name (_is_reserved_catalog_name rejects creating/renaming a
            # server to either; the local branch's library_keys skip hides
            # user servers matching either key) -- and the display name below is
            # deliberately kept as the friendlier "Chrome", so a user-created
            # server literally named "chrome"/"Chrome" is still rejected on
            # create/rename and suppressed from the local listing. Note the two
            # halves differ in reach: create/rename rejection is prospective,
            # while the listing skip is retroactive, so a row named after this
            # id or name that predates either guard is hidden from the picker's
            # Local tab (never from /api/mcp/servers) once it lands. Product
            # chose the display name over freeing that (plausible but niche)
            # name; scoping the id at least keeps the shared server row and
            # connect-flow naming collision-free.
            "app_id": "chrome-devtools",
            "name": "Chrome",
            "description": "Automate a Chrome browser: open pages, read content, fill forms, take screenshots, and inspect network requests.",
            "icon": "https://www.google.com/s2/favicons?domain=chrome.google.com&sz=128",
            "transport": "stdio",
            "provider_name": None,
            "category": "Productivity",
            "oauth_scopes": None,
            # Hidden until the runtime supports execution-scoped (persistent)
            # stdio MCP sessions: today every tool call spawns a fresh
            # chrome-devtools-mcp process (mcp_adapter._execute_mcp_call ->
            # create_session), so browser/page state does not survive across
            # calls and any multi-step flow (navigate -> click/fill) breaks.
            # Once persistent sessions land, admins can re-enable via
            # PATCH /api/admin/mcp/apps — is_visible_in_connector is not a
            # builtin-protected field, so no redeploy is needed.
            "is_visible_in_connector": False,
            # Keyless (non-oauth): no secrets to collect — connecting only
            # creates the per-user association via POST /api/mcp/apps/{id}/connect.
            # --headless/--isolated keep server deployments displayless and give
            # each session a throwaway profile instead of a shared one.
            # Version-pinned: npx resolves this on every launch, so an
            # unpinned tag would silently pick up new upstream releases;
            # both Dockerfile.backend (non-sandboxed launches) and
            # Dockerfile.sandbox (should_sandbox_mcp_connection() routes
            # every npx/uvx MCP connector into the sandbox image whenever
            # one is configured) independently pre-install this exact
            # version and warm an npx cache for it. That warm-up is
            # intended to make npx resolve locally instead of hitting the
            # npm registry on every launch, but currently does not: the
            # MCP stdio launcher this connector's process goes through
            # only forwards a fixed env allowlist to the spawned npx
            # child (no NPM_CONFIG_CACHE), so every launch still needs
            # npm-registry access regardless of the warm-up -- tracked in
            # xorbitsai/xagent#1869, not yet fixed here.
            # No --executablePath/--channel: the default "stable" channel
            # resolution finds Chrome per-platform (/Applications/... on
            # macOS dev hosts, /opt/google/chrome/chrome in both the
            # backend and sandbox images, which each guarantee that path
            # exists) — a hardcoded path here would break every other
            # platform.
            # --chrome-arg='--no-sandbox'/'--disable-setuid-sandbox': both
            # the backend and sandbox containers run Chrome as root,
            # needing the same root/no-sandbox exposure the existing
            # browser_use tool (core/tools/core/browser_use.py) already
            # carries in the backend image -- not identical flags
            # (browser_use passes only --no-sandbox, plus
            # --disable-web-security and file-access flags this connector
            # does not set), just a comparable, already-accepted
            # trade-off, not a new class of risk.
            # --chrome-arg='--disable-dev-shm-usage': Docker's default
            # /dev/shm size (64MB) is a well-documented Chrome-in-container
            # crash mode: --no-sandbox is set above regardless, so this
            # costs nothing and heads off a real failure mode once this
            # connector is enabled, rather than waiting to hit it live.
            # --no-usage-statistics/--no-performance-crux: chrome-devtools-mcp
            # sends usage telemetry to Google and POSTs page URLs to the CrUX
            # API by default; opt out until this connector's data flow is
            # disclosed to users.
            "launch_config": {
                "command": "npx",
                "args": [
                    "-y",
                    # npm-exec flag, must precede the package spec: intended
                    # to let the exact-version cache warmed at image build
                    # time (both Dockerfile.backend and Dockerfile.sandbox)
                    # skip the npm registry at launch, though that warm-up
                    # currently doesn't reach this connector's actual npx
                    # process (xorbitsai/xagent#1869) -- harmless either
                    # way, since --prefer-offline still falls back to a
                    # normal fetch on any cache miss.
                    "--prefer-offline",
                    "chrome-devtools-mcp@1.6.0",
                    "--headless",
                    "--isolated",
                    "--chrome-arg=--no-sandbox",
                    "--chrome-arg=--disable-setuid-sandbox",
                    "--chrome-arg=--disable-dev-shm-usage",
                    "--no-usage-statistics",
                    "--no-performance-crux",
                ],
            },
        },
        {
            "app_id": "github",
            "name": "GitHub",
            # Explicitly discloses the "repo" scope's blast radius (all
            # repositories -- public and private -- the connecting account
            # can access, not just ones the user picks) rather than leaving
            # it implicit, since this is an OAuth App with no per-repository
            # allowlist (unlike a GitHub App's repository-selection step).
            "description": "Connect to GitHub to search repositories and code, read and create issues and pull requests, comment, and browse file contents and commit history. Grants access to every repository (public and private) the connected account can access -- there is no per-repository selection.",
            "icon": "https://www.google.com/s2/favicons?domain=github.com&sz=128",
            "transport": "oauth",
            "provider_name": "github",
            "category": "Development",
            # read:org dropped: no tool here calls any organization endpoint
            # (no org listing/membership lookup), so it was requested
            # without a corresponding capability -- principle of least
            # privilege. user:email is kept: unlike a separate unused
            # endpoint, it changes what /user's own "email" field returns
            # (populates it for accounts without a public email), which
            # github_get_current_user's output relies on directly -- see
            # test_get_current_user_returns_profile in test_github_mcp.py,
            # which asserts the email field is surfaced, and
            # test_github_login_requests_exact_canonical_scope in
            # test_generic_oauth_login.py, which asserts this exact scope
            # is what gets requested at authorize time.
            "oauth_scopes": ["repo", "user:email"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.github"],
                "env_mapping": {"GITHUB_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "intercom",
            "name": "Intercom",
            "description": "Connect to Intercom to search contacts, review conversations, and reply to customers.",
            "icon": "https://www.google.com/s2/favicons?domain=intercom.com&sz=128",
            "transport": "oauth",
            "provider_name": "intercom",
            "category": "Support",
            "oauth_scopes": [],
            # Hidden until manually verified against a live workspace (all
            # three regions, ideally), same precedent as the chrome row
            # above: it ships customer-facing *write* tools (reply/note/
            # close) discoverable by every user the moment it's visible, and
            # that hasn't happened yet. Flip to True via a follow-up once
            # that verification is done -- no redeploy needed, per the chrome
            # row's note (is_visible_in_connector is not builtin-protected).
            "is_visible_in_connector": False,
            # A local FastMCP module talking to Intercom's REST API directly,
            # not Intercom's hosted MCP server (mcp.intercom.com) -- that
            # hosted server does not support AU-hosted workspaces, while the
            # underlying REST API and OAuth (api.intercom.io, auto-routed)
            # do, which this connector needs for AU customers. See the
            # oauth_providers row above for the region-coverage citation.
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.intercom"],
                "env_mapping": {"INTERCOM_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "salesforce",
            "name": "Salesforce",
            "description": "Connect to Salesforce to query and manage records (accounts, contacts, leads, opportunities, and custom objects) with SOQL/SOSL, and browse object schemas.",
            "icon": "https://www.google.com/s2/favicons?domain=salesforce.com&sz=128",
            "transport": "oauth",
            "provider_name": "salesforce",
            "category": "CRM",
            "oauth_scopes": ["api", "refresh_token", "openid"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.salesforce"],
                # instance_url is the per-org API host from the OAuth grant
                # (UserOAuth.instance_url) -- see the oauth_providers row
                # above for why Salesforce needs this second mapping entry
                # that no other connector here does.
                "env_mapping": {
                    "SALESFORCE_ACCESS_TOKEN": "access_token",
                    "SALESFORCE_INSTANCE_URL": "instance_url",
                },
            },
        },
        {
            "app_id": "deputy",
            "name": "Deputy",
            "description": "Connect to Deputy to look up employees, view rosters/shifts, and read timesheets.",
            "icon": "https://www.google.com/s2/favicons?domain=deputy.com&sz=128",
            "transport": "oauth",
            "provider_name": "deputy",
            "category": "Scheduling",
            "oauth_scopes": ["longlife_refresh_token"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.deputy"],
                # instance_url is the per-install API host from the OAuth
                # grant (UserOAuth.instance_url) -- see the oauth_providers
                # row above for why Deputy needs this second mapping entry,
                # the same reason Salesforce's row above does.
                "env_mapping": {
                    "DEPUTY_ACCESS_TOKEN": "access_token",
                    "DEPUTY_INSTANCE_URL": "instance_url",
                },
            },
        },
        {
            "app_id": "linear",
            "name": "Linear",
            "description": "Connect to Linear to search and manage issues, comments, teams, and projects.",
            "icon": "https://www.google.com/s2/favicons?domain=linear.app&sz=128",
            "transport": "oauth",
            "provider_name": "linear",
            "category": "Productivity",
            "oauth_scopes": ["read", "write"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.linear"],
                "env_mapping": {"LINEAR_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "jira",
            "name": "Jira",
            "description": "Connect to Jira to search issues with JQL, read and create issues and comments, transition issues through their workflow, and browse projects and users.",
            "icon": "https://www.google.com/s2/favicons?domain=atlassian.com&sz=128",
            "transport": "oauth",
            "provider_name": "jira",
            "category": "Productivity",
            "oauth_scopes": [
                "offline_access",
                "read:jira-work",
                "write:jira-work",
                "read:jira-user",
                "read:me",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.jira"],
                "env_mapping": {"JIRA_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "stripe",
            "name": "Stripe",
            "description": "Connect to Stripe with a Restricted API Key (not your full secret key) to look up customers, charges, invoices, and subscriptions, and issue refunds.",
            "icon": "https://www.google.com/s2/favicons?domain=stripe.com&sz=128",
            "transport": "stdio",
            "provider_name": None,
            "category": "Payments",
            "oauth_scopes": None,
            "is_visible_in_connector": True,
            # Key-based (non-oauth), like aws/google-maps/posthog: Stripe's
            # own OAuth flow (Connect "Standard accounts") is for a platform
            # aggregating many *other* merchants' accounts, not for a single
            # user granting a tool access to their own account -- Stripe's
            # docs explicitly discourage it for new integrations, and since
            # June 2021 a read_write OAuth connection fails outright if the
            # account is already connected to another platform. A
            # Restricted API Key (rk_live_.../rk_test_...) sidesteps both
            # problems entirely and is what Stripe's own agent-toolkit/MCP
            # server recommends for exactly this use case (see
            # github.com/stripe/agent-toolkit tools/README.md) -- same
            # no-review self-serve bar as an OAuth App, scoped to only the
            # permissions the user grants it.
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.stripe"],
                "required_env": ["STRIPE_API_KEY"],
            },
        },
        {
            "app_id": "employment-hero",
            "name": "Employment Hero",
            "description": "Connect to Employment Hero to look up organisations, employees, teams, and timesheet entries.",
            "icon": "https://www.google.com/s2/favicons?domain=employmenthero.com&sz=128",
            "transport": "oauth",
            "provider_name": "employment-hero",
            # No existing sidebar category filter (connect-mcp-dialog.tsx)
            # fits an HR/payroll connector -- reusing "Operations" (currently
            # only aws, an infra-monitoring connector) would be the same
            # mismatch the salesforce/notion comments above call out on
            # other rows, not something to repeat. Like storage/development/
            # productivity elsewhere in this file, this category has no
            # dedicated sidebar button; the connector still surfaces under
            # "All".
            "category": "HR",
            "oauth_scopes": [
                "organisations:list",
                "employees:list",
                "employees:show",
                "teams:list",
                "timesheet_entries:list",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.employment_hero"],
                "env_mapping": {"EMPLOYMENT_HERO_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "chartmogul",
            "name": "ChartMogul",
            "description": "Connect your ChartMogul account (with a per-user API key from Profile -> API keys) to look up and manage customers, contacts, and sales opportunities.",
            "icon": "https://www.google.com/s2/favicons?domain=chartmogul.com&sz=128",
            "transport": "stdio",
            "provider_name": None,
            # CRM, not Analytics: the 15 tools this connector exposes are
            # exclusively customer/contact/opportunity CRUD -- no metrics,
            # subscription, MRR, or churn endpoints, despite ChartMogul the
            # product being analytics-branded.
            "category": "CRM",
            "oauth_scopes": None,
            "is_visible_in_connector": True,
            # Key-based (non-oauth), like aws/google-maps/posthog/stripe:
            # ChartMogul has no OAuth flow at all, only a per-user API key
            # generated from Profile -> API keys in their own account --
            # sent as HTTP Basic Auth username (confirmed against
            # chartmogul-python's Config class), same no-review self-serve
            # bar as an OAuth App.
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.chartmogul"],
                "required_env": ["CHARTMOGUL_API_KEY"],
            },
        },
    ]


_BUILTIN_EXECUTION_FIELD_NAMES = (
    "name",
    "transport",
    "provider_name",
    "oauth_scopes",
    "launch_config",
)


def get_builtin_public_mcp_app(app_id: str) -> dict[str, Any] | None:
    for row in get_builtin_public_mcp_app_rows():
        if row["app_id"] == app_id:
            return deepcopy(row)
    return None


def is_builtin_public_mcp_app(app_id: str) -> bool:
    return any(row["app_id"] == app_id for row in get_builtin_public_mcp_app_rows())


def get_builtin_execution_fields(app_id: str) -> dict[str, Any] | None:
    row = get_builtin_public_mcp_app(app_id)
    if row is None:
        return None
    return deepcopy(
        {field_name: row[field_name] for field_name in _BUILTIN_EXECUTION_FIELD_NAMES}
    )


def get_builtin_execution_fields_and_optional_scopes(
    app_id: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """The app's execution fields, plus OAuth scopes requested via the
    authorize request's optional_scope parameter rather than its required
    scope parameter (see api/auth.py).

    A single registry scan (plus one deepcopy of the matched row) serving
    both needs: _app_to_dict wants both for every app on a listing path,
    and scanning + deepcopying the full row twice per app to get them
    separately would be wasted work.

    optional_oauth_scopes is deliberately not in
    _BUILTIN_EXECUTION_FIELD_NAMES: unlike oauth_scopes, there is no legacy
    persisted-row state to drift-check against for a field this new, so
    this reads straight from the row rather than going through the
    DB-column drift-tracking machinery. Most builtin apps have no optional
    scopes at all, hence the plain ``.get`` default.
    """
    row = get_builtin_public_mcp_app(app_id)
    if row is None:
        return None, []
    execution_fields = {
        field_name: row[field_name] for field_name in _BUILTIN_EXECUTION_FIELD_NAMES
    }
    return execution_fields, list(row.get("optional_oauth_scopes", []))


def _safe_configuration_hash(values: dict[str, Any]) -> str:
    serialized = json.dumps(
        values,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def validate_builtin_public_mcp_apps(bind: Connection) -> list[dict[str, Any]]:
    """Return safe summaries for persisted built-in catalog drift.

    Missing rows are intentionally outside this drift-only check; supported
    admin writes cannot delete built-ins, and this validator never recreates
    data. Only rows selected by exact built-in ``app_id`` are compared, so
    custom applications remain database-owned.
    """

    canonical_rows = get_builtin_public_mcp_app_rows()
    canonical_by_app_id = {row["app_id"]: row for row in canonical_rows}
    selected_columns = [
        PUBLIC_MCP_APPS_TABLE.c.app_id,
        *(PUBLIC_MCP_APPS_TABLE.c[field] for field in _BUILTIN_EXECUTION_FIELD_NAMES),
    ]
    persisted_rows = bind.execute(
        sa.select(*selected_columns).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id.in_(tuple(canonical_by_app_id))
        )
    ).mappings()
    persisted_by_app_id = {row["app_id"]: row for row in persisted_rows}

    mismatches: list[dict[str, Any]] = []
    for app_id, canonical_row in canonical_by_app_id.items():
        persisted_row = persisted_by_app_id.get(app_id)
        if persisted_row is None:
            continue

        mismatched_fields = [
            field_name
            for field_name in _BUILTIN_EXECUTION_FIELD_NAMES
            if persisted_row[field_name] != canonical_row[field_name]
        ]
        if not mismatched_fields:
            continue

        canonical_values = {
            field_name: canonical_row[field_name] for field_name in mismatched_fields
        }
        persisted_values = {
            field_name: persisted_row[field_name] for field_name in mismatched_fields
        }
        mismatches.append(
            {
                "app_id": app_id,
                "mismatched_fields": mismatched_fields,
                "canonical_hash": _safe_configuration_hash(canonical_values),
                "persisted_hash": _safe_configuration_hash(persisted_values),
            }
        )

    return mismatches


def _filter_row(row: dict[str, Any], allowed_columns: set[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def seed_builtin_oauth_and_public_mcp_apps(bind: Connection) -> None:
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "oauth_providers" in existing_tables:
        oauth_columns = {
            column["name"] for column in inspector.get_columns("oauth_providers")
        }
        existing_provider_names = set(
            bind.execute(sa.select(OAUTH_PROVIDERS_TABLE.c.provider_name)).scalars()
        )
        provider_rows_to_insert = [
            _filter_row(row, oauth_columns)
            for row in get_builtin_oauth_provider_rows()
            if row["provider_name"] not in existing_provider_names
        ]
        if provider_rows_to_insert:
            bind.execute(sa.insert(OAUTH_PROVIDERS_TABLE), provider_rows_to_insert)

    if "public_mcp_apps" in existing_tables:
        app_columns = {
            column["name"] for column in inspector.get_columns("public_mcp_apps")
        }
        # Same guard as the chrome seed migration's upgrade(), and for the
        # same reason: is_visible_in_connector is load-bearing for the
        # hidden-rollout gate several rows ship behind (chrome today), so
        # _filter_row silently dropping it and falling back to the table's
        # visible-by-default would seed those rows visible. This caller only
        # runs against a table just created from the current model (see
        # should_seed_builtin_mcp_registry in database.py), so the column
        # should always be present here -- this is belt-and-suspenders, not
        # a path expected to actually fire.
        if "is_visible_in_connector" not in app_columns:
            raise RuntimeError(
                "public_mcp_apps.is_visible_in_connector is missing; "
                "builtin apps that ship hidden must not seed visible"
            )
        existing_app_ids = set(
            bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars()
        )
        app_rows_to_insert = [
            _filter_row(row, app_columns)
            for row in get_builtin_public_mcp_app_rows()
            if row["app_id"] not in existing_app_ids
        ]
        if app_rows_to_insert:
            bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), app_rows_to_insert)
