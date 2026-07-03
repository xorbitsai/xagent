"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { ArrowLeft } from "lucide-react";
import { useI18n } from "@/contexts/i18n-context";
import { getBrandingFromEnv } from "@/lib/branding";
import { LEGAL_CONTACT_EMAIL } from "@/lib/legal";

const branding = getBrandingFromEnv();

const CONTACT_EMAIL = LEGAL_CONTACT_EMAIL;
const LAST_UPDATED = "2026-07-02";

function H2({ children }: { children: ReactNode }) {
  return <h2 className="text-lg font-bold text-foreground mt-8 mb-2">{children}</h2>;
}

function P({ children }: { children: ReactNode }) {
  return <p className="text-[15px] leading-7 text-foreground/90 mb-3">{children}</p>;
}

function Ul({ children }: { children: ReactNode }) {
  return <ul className="list-disc pl-6 space-y-2 text-[15px] leading-7 text-foreground/90 mb-3">{children}</ul>;
}

function A({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary underline hover:no-underline">
      {children}
    </a>
  );
}

function MailA({ children }: { children: ReactNode }) {
  return (
    <a href={`mailto:${CONTACT_EMAIL}`} className="text-primary underline hover:no-underline">
      {children}
    </a>
  );
}

function EnglishContent() {
  return (
    <>
      <P>
        This Privacy Policy explains how {branding.appName} (&quot;{branding.appName}&quot;,
        &quot;we&quot;, &quot;us&quot;, or &quot;our&quot;) collects, uses, stores, and
        protects information when you use our website and services (the &quot;Service&quot;).
      </P>

      <H2>1. Information We Collect</H2>
      <P>We collect the following categories of information:</P>
      <Ul>
        <li>
          <strong>Account information</strong>: name, email address, and password
          (encrypted) when you register for an account.
        </li>
        <li>
          <strong>Content you provide</strong>: tasks, prompts, files, and other content
          you submit to agents you configure within the Service.
        </li>
        <li>
          <strong>Google user data</strong>: if you choose to connect a Google account
          (e.g. Gmail, Calendar, Drive) to an agent, we access the data necessary to
          perform the actions you configure, such as reading, sending, or organizing
          email messages, or reading and creating calendar events and files, only for
          the scopes you explicitly authorize via Google&apos;s OAuth consent screen.
        </li>
        <li>
          <strong>Usage data</strong>: log data, device information, and diagnostic
          data collected automatically to operate and improve the Service.
        </li>
      </Ul>

      <H2>2. How We Use Information</H2>
      <P>We use the information we collect solely to:</P>
      <Ul>
        <li>Provide, operate, and maintain the Service and the agents you configure;</li>
        <li>
          Perform the specific actions you request (e.g. drafting or sending an email
          on your behalf, retrieving a file, scheduling an event);
        </li>
        <li>Authenticate you and secure your account;</li>
        <li>Diagnose technical issues and improve reliability and performance;</li>
        <li>Communicate with you about your account or the Service.</li>
      </Ul>
      <P>
        We do not use the information you provide, including Google user data, to
        serve advertisements, and we do not sell your data to third parties.
      </P>

      <H2>3. Google User Data &amp; Limited Use Disclosure</H2>
      <P>
        {branding.appName}&apos;s use and transfer to any other app of information
        received from Google APIs will adhere to the{" "}
        <A href="https://developers.google.com/terms/api-services-user-data-policy">
          Google API Services User Data Policy
        </A>
        , including the Limited Use requirements.
      </P>
      <P>Specifically:</P>
      <Ul>
        <li>
          We only request the Google scopes required to perform the actions you
          explicitly configure your agent to take (for example, reading or sending
          email, or managing calendar/drive files you designate).
        </li>
        <li>
          Google user data is used exclusively to provide user-facing features within
          the Service that you have requested, and is never used to develop, improve,
          or train generalized AI/ML models unrelated to your direct use of the
          Service.
        </li>
        <li>
          We do not transfer Google user data to third parties except as necessary to
          provide the Service (e.g. to our infrastructure/hosting providers acting on
          our behalf, under confidentiality obligations), to comply with applicable
          law, or as part of a merger or acquisition involving our business, with
          notice to you.
        </li>
        <li>
          We do not allow humans to read Google user data unless: (a) we have your
          affirmative agreement for specific messages/files, (b) it is necessary for
          security purposes (e.g. investigating abuse), (c) it is necessary to comply
          with applicable law, or (d) it is aggregated and anonymized and used
          strictly for internal operations in accordance with Google&apos;s policies.
        </li>
      </Ul>

      <H2>4. Data Storage, Retention &amp; Security</H2>
      <P>
        OAuth tokens and other sensitive credentials are encrypted at rest. Data is
        stored on infrastructure with access controls restricted to personnel who need
        it to operate the Service. We retain your data only for as long as necessary
        to provide the Service or as required by law, and delete it upon account
        deletion or upon your request, subject to reasonable backup retention periods.
      </P>

      <H2>5. Your Rights &amp; Choices</H2>
      <Ul>
        <li>
          You can revoke {branding.appName}&apos;s access to your Google account at
          any time from your{" "}
          <A href="https://myaccount.google.com/permissions">Google Account permissions page</A>.
        </li>
        <li>
          You can request access to, correction of, or deletion of your personal data
          by contacting us at <MailA>{CONTACT_EMAIL}</MailA>.
        </li>
        <li>You can delete your account and associated data from your account settings.</li>
      </Ul>

      <H2>6. Children&apos;s Privacy</H2>
      <P>
        The Service is not directed to individuals under the age of 16, and we do not
        knowingly collect personal information from children.
      </P>

      <H2>7. Changes to This Policy</H2>
      <P>
        We may update this Privacy Policy from time to time. We will notify you of
        material changes by posting the updated policy on this page with a new
        &quot;last updated&quot; date.
      </P>

      <H2>8. Contact Us</H2>
      <P>
        If you have any questions about this Privacy Policy, please contact us at{" "}
        <MailA>{CONTACT_EMAIL}</MailA>.
      </P>
    </>
  );
}

function ChineseContent() {
  return (
    <>
      <P>
        本隐私政策说明 {branding.appName}（以下简称&quot;我们&quot;）在您使用我们的网站及相关服务（以下简称&quot;服务&quot;）时，如何收集、使用、存储和保护相关信息。
      </P>

      <H2>一、我们收集的信息</H2>
      <P>我们会收集以下类别的信息：</P>
      <Ul>
        <li><strong>账户信息</strong>：注册账户时提供的姓名、电子邮箱地址及（加密存储的）密码。</li>
        <li><strong>您提供的内容</strong>：您在服务中配置智能体（Agent）时提交的任务、提示词、文件等内容。</li>
        <li>
          <strong>Google 用户数据</strong>：如果您选择将 Google 账号（例如 Gmail、日历、云端硬盘）连接到智能体，我们只会在您通过
          Google OAuth 授权页面明确同意的权限范围内，访问执行您所配置操作所必需的数据，例如读取/发送邮件，或读取/创建日历事件与文件。
        </li>
        <li><strong>使用数据</strong>：为运行和改进服务而自动收集的日志数据、设备信息及诊断数据。</li>
      </Ul>

      <H2>二、我们如何使用信息</H2>
      <P>我们收集的信息仅用于：</P>
      <Ul>
        <li>提供、运行和维护本服务及您所配置的智能体；</li>
        <li>执行您请求的具体操作（例如代您起草或发送邮件、获取文件、创建日程）；</li>
        <li>对您进行身份验证并保障账户安全；</li>
        <li>诊断技术问题、提升服务稳定性与性能；</li>
        <li>就您的账户或本服务与您沟通。</li>
      </Ul>
      <P>我们不会使用您提供的信息（包括 Google 用户数据）用于投放广告，也不会将您的数据出售给第三方。</P>

      <H2>三、Google 用户数据与&quot;有限使用&quot;声明</H2>
      <P>
        {branding.appName} 对通过 Google API 获取的信息的使用及向其他应用的转移，将遵循{" "}
        <A href="https://developers.google.com/terms/api-services-user-data-policy">
          Google API 服务用户数据政策
        </A>
        ，包括其中的&quot;有限使用&quot;（Limited Use）要求。
      </P>
      <P>具体而言：</P>
      <Ul>
        <li>我们仅申请执行您明确配置的智能体操作所必需的 Google 权限范围（例如读取/发送邮件，或管理您指定的日历/云端硬盘文件）。</li>
        <li>Google 用户数据仅用于向您提供您在服务中主动请求的面向用户的功能，绝不会用于开发、改进或训练与您直接使用本服务无关的通用人工智能/机器学习模型。</li>
        <li>除非为提供本服务所必需（例如交由承担保密义务、代表我们运作的基础设施/托管服务提供商处理）、为遵守适用法律，或作为涉及我方业务的合并或收购的一部分并事先通知您，我们不会向第三方转移 Google 用户数据。</li>
        <li>
          除非满足以下情形之一，我们不允许任何人工人员查看 Google 用户数据：（a）获得您对特定邮件/文件的明确同意；（b）出于安全目的所必需（例如调查滥用行为）；（c）为遵守适用法律所必需；或（d）数据已聚合且匿名化，并严格按照
          Google 政策仅用于内部运营。
        </li>
      </Ul>

      <H2>四、数据存储、保留与安全</H2>
      <P>
        OAuth 令牌及其他敏感凭证在静态存储时均经过加密。数据存储在访问权限受限的基础设施上，仅限运行本服务所必需的人员访问。我们仅在提供服务所必需或法律要求的期限内保留您的数据，并在您注销账户或提出请求后删除数据（合理的备份保留期除外）。
      </P>

      <H2>五、您的权利与选择</H2>
      <Ul>
        <li>
          您可以随时通过{" "}
          <A href="https://myaccount.google.com/permissions">Google 账号权限管理页面</A>
          撤销 {branding.appName} 对您 Google 账号的访问权限。
        </li>
        <li>您可以通过 <MailA>{CONTACT_EMAIL}</MailA> 联系我们，申请访问、更正或删除您的个人数据。</li>
        <li>您可以在账户设置中删除您的账户及相关数据。</li>
      </Ul>

      <H2>六、儿童隐私</H2>
      <P>本服务不面向 16 周岁以下的未成年人，我们不会在知情的情况下收集儿童的个人信息。</P>

      <H2>七、本政策的变更</H2>
      <P>我们可能不时更新本隐私政策。如有重大变更，我们会在本页面发布更新后的政策并更新&quot;最后更新日期&quot;。</P>

      <H2>八、联系我们</H2>
      <P>如您对本隐私政策有任何疑问，请通过 <MailA>{CONTACT_EMAIL}</MailA> 与我们联系。</P>
    </>
  );
}

export default function PrivacyPolicyPage() {
  const { locale, t } = useI18n();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-3xl mx-auto px-6 py-16">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-primary transition-colors mb-8"
        >
          <ArrowLeft className="w-4 h-4" />
          {t("common.back")}
        </Link>

        <h1 className="text-3xl font-bold mb-2">{t("footer.privacyPolicy")}</h1>
        <p className="text-sm text-muted-foreground mb-10">
          {locale === "zh" ? `最后更新日期：${LAST_UPDATED}` : `Last updated: ${LAST_UPDATED}`}
        </p>

        {locale === "zh" ? <ChineseContent /> : <EnglishContent />}
      </div>
    </div>
  );
}
