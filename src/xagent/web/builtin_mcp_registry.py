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
            "oauth_scopes": ["https://www.googleapis.com/auth/calendar"],
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
            "description": "Connect to Slack to search and read channel, thread, and DM history, post messages and replies, react to messages, and upload files, e.g. incident summaries and recommended fixes.",
            "icon": "https://www.google.com/s2/favicons?domain=slack.com&sz=128",
            "transport": "oauth",
            "provider_name": "slack",
            "category": "Communication",
            "oauth_scopes": [
                "chat:write",
                "chat:write.public",
                "channels:read",
                "channels:history",
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
            # "chrome-devtools", not the generic "chrome", matching the
            # upstream package name and the vendor-scoped ids every other
            # seeded app uses. Known, accepted caveat: api/mcp.py couples
            # catalog apps to user-server names by normalized id AND display
            # name (_is_reserved_catalog_name rejects creating/renaming a
            # server to either; library_names hides user servers whose name
            # matches a catalog app name) -- and the display name below is
            # deliberately kept as the friendlier "Chrome", so a user-created
            # server literally named "chrome"/"Chrome" is still rejected on
            # create/rename and suppressed from the local listing. Product
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
            # unpinned tag would silently pick up new upstream releases; the
            # backend image (Dockerfile.backend) pre-installs this exact
            # version globally so npx resolves it locally instead of hitting
            # the npm registry on every server launch.
            # No --executablePath/--channel: the default "stable" channel
            # resolution finds Chrome per-platform (/Applications/... on
            # macOS dev hosts, /opt/google/chrome/chrome in the backend
            # image, which Dockerfile.backend guarantees exists) — a
            # hardcoded path here would break every other platform.
            # --chrome-arg='--no-sandbox'/'--disable-setuid-sandbox': the
            # backend container runs Chrome as root, needing the same root/
            # no-sandbox exposure the existing browser_use tool
            # (core/tools/core/browser_use.py) already carries in this image
            # -- not identical flags (browser_use passes only --no-sandbox,
            # plus --disable-web-security and file-access flags this
            # connector does not set), just a comparable, already-accepted
            # trade-off, not a new class of risk.
            # --no-usage-statistics/--no-performance-crux: chrome-devtools-mcp
            # sends usage telemetry to Google and POSTs page URLs to the CrUX
            # API by default; opt out until this connector's data flow is
            # disclosed to users.
            "launch_config": {
                "command": "npx",
                "args": [
                    "-y",
                    # npm-exec flag, must precede the package spec: with the
                    # exact-version cache warmed at image build time
                    # (Dockerfile.backend), resolution never hits the npm
                    # registry at launch; on a cache miss (e.g. first run on
                    # a dev machine) it still fetches normally.
                    "--prefer-offline",
                    "chrome-devtools-mcp@1.6.0",
                    "--headless",
                    "--isolated",
                    "--chrome-arg=--no-sandbox",
                    "--chrome-arg=--disable-setuid-sandbox",
                    "--no-usage-statistics",
                    "--no-performance-crux",
                ],
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
