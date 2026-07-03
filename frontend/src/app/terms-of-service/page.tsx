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
        These Terms of Service (&quot;Terms&quot;) govern your access to and use of{" "}
        {branding.appName} (&quot;{branding.appName}&quot;, &quot;we&quot;, &quot;us&quot;, or
        &quot;our&quot;) and its associated website and services (the &quot;Service&quot;). By
        creating an account or otherwise using the Service, you agree to be bound by these
        Terms.
      </P>

      <H2>1. The Service</H2>
      <P>
        {branding.appName} lets you build and run AI-powered agents that can, at your
        direction, use tools and connected third-party accounts (such as Google Gmail,
        Calendar, or Drive) to complete tasks on your behalf. You are responsible for the
        agents you configure and the actions you authorize them to take.
      </P>

      <H2>2. Accounts</H2>
      <Ul>
        <li>You must provide accurate information when creating an account and keep your credentials secure.</li>
        <li>You are responsible for all activity that occurs under your account.</li>
        <li>You must notify us promptly of any unauthorized use of your account.</li>
      </Ul>

      <H2>3. Third-Party Connections (Including Google APIs)</H2>
      <P>
        The Service may allow you to connect third-party accounts, including Google
        services, via OAuth. When you do so, you authorize the Service to access the data
        and perform the actions covered by the specific permissions (scopes) you approve.
        Your use of connected Google services is also subject to Google&apos;s own terms and
        policies. You may revoke access at any time from the relevant third-party
        account&apos;s security settings. Our handling of Google user data is further
        described in our{" "}
        <Link href="/privacy-policy" className="text-primary underline hover:no-underline">
          Privacy Policy
        </Link>
        .
      </P>

      <H2>4. Acceptable Use</H2>
      <P>You agree not to use the Service to:</P>
      <Ul>
        <li>Violate any applicable law or regulation;</li>
        <li>Send unsolicited bulk email (spam) or engage in phishing, fraud, or harassment;</li>
        <li>Access or attempt to access accounts, data, or systems without authorization;</li>
        <li>Reverse engineer, disrupt, or interfere with the integrity or performance of the Service;</li>
        <li>Upload or transmit malicious code.</li>
      </Ul>

      <H2>5. Content &amp; Intellectual Property</H2>
      <P>
        You retain ownership of the content you submit to the Service. You grant us a
        limited license to process that content solely to operate and provide the Service
        to you. The Service, including its software and branding, is owned by us or our
        licensors and is protected by intellectual property laws.
      </P>

      <H2>6. Disclaimers</H2>
      <P>
        The Service is provided &quot;as is&quot; and &quot;as available&quot; without
        warranties of any kind, express or implied. Agents act on instructions you provide
        and may make mistakes; you are responsible for reviewing agent actions before
        relying on them, particularly actions that send communications or modify data on
        your behalf.
      </P>

      <H2>7. Limitation of Liability</H2>
      <P>
        To the maximum extent permitted by law, {branding.appName} shall not be liable for
        any indirect, incidental, special, consequential, or punitive damages, or any loss
        of data, revenue, or profits, arising out of or related to your use of the Service.
      </P>

      <H2>8. Termination</H2>
      <P>
        We may suspend or terminate your access to the Service if you violate these Terms.
        You may stop using the Service and delete your account at any time from your
        account settings.
      </P>

      <H2>9. Changes to These Terms</H2>
      <P>
        We may update these Terms from time to time. Material changes will be communicated
        by posting the updated Terms on this page with a new &quot;last updated&quot; date.
        Continued use of the Service after changes take effect constitutes acceptance of
        the revised Terms.
      </P>

      <H2>10. Contact Us</H2>
      <P>
        If you have any questions about these Terms, please contact us at{" "}
        <MailA>{CONTACT_EMAIL}</MailA>.
      </P>
    </>
  );
}

function ChineseContent() {
  return (
    <>
      <P>
        本服务条款（以下简称&quot;本条款&quot;）适用于您访问和使用 {branding.appName}
        （以下简称&quot;我们&quot;）及其相关网站和服务（以下简称&quot;服务&quot;）。创建账户或以其他方式使用本服务，即表示您同意受本条款约束。
      </P>

      <H2>一、服务说明</H2>
      <P>
        {branding.appName} 允许您构建并运行由人工智能驱动的智能体（Agent），该智能体可以根据您的指示，使用各类工具及您连接的第三方账户（如
        Google Gmail、日历、云端硬盘）代您完成任务。您需对自己配置的智能体及授权其执行的操作负责。
      </P>

      <H2>二、账户</H2>
      <Ul>
        <li>创建账户时您须提供真实信息，并妥善保管账户凭证。</li>
        <li>您需对您账户下发生的所有活动负责。</li>
        <li>如发现账户被未经授权使用，请及时通知我们。</li>
      </Ul>

      <H2>三、第三方连接（含 Google API）</H2>
      <P>
        本服务允许您通过 OAuth 连接第三方账户，包括 Google 相关服务。连接后，即表示您授权本服务在您所批准的具体权限（scope）范围内访问相关数据并执行相应操作。您对已连接
        Google 服务的使用同样受 Google 自身条款和政策的约束，您可以随时在相应第三方账户的安全设置中撤销授权。我们对 Google 用户数据的处理方式详见我们的
        <Link href="/privacy-policy" className="text-primary underline hover:no-underline">
          隐私政策
        </Link>
        。
      </P>

      <H2>四、可接受使用</H2>
      <P>您同意不会将本服务用于：</P>
      <Ul>
        <li>违反任何适用法律法规；</li>
        <li>发送未经请求的批量邮件（垃圾邮件），或从事钓鱼、欺诈或骚扰行为；</li>
        <li>未经授权访问或试图访问他人账户、数据或系统；</li>
        <li>对本服务进行逆向工程，或干扰、破坏本服务的完整性与性能；</li>
        <li>上传或传播恶意代码。</li>
      </Ul>

      <H2>五、内容与知识产权</H2>
      <P>
        您对提交到本服务的内容保留所有权。您授予我们有限的许可，仅用于处理该内容以运行并向您提供本服务。本服务（包括其软件与品牌标识）归我们或我们的许可方所有，并受知识产权法保护。
      </P>

      <H2>六、免责声明</H2>
      <P>
        本服务按&quot;现状&quot;及&quot;可用性&quot;提供，不附带任何明示或暗示的保证。智能体依据您提供的指示执行操作，可能出现错误；在依赖智能体执行结果之前，尤其是涉及代您发送通讯或修改数据的操作，您有责任先行核对。
      </P>

      <H2>七、责任限制</H2>
      <P>
        在法律允许的最大范围内，{branding.appName} 不对因使用本服务而产生或与之相关的任何间接、附带、特殊、后果性或惩罚性损害，或数据、收入、利润的损失承担责任。
      </P>

      <H2>八、终止</H2>
      <P>
        如您违反本条款，我们可暂停或终止您对本服务的访问权限。您也可以随时停止使用本服务，并在账户设置中删除您的账户。
      </P>

      <H2>九、条款变更</H2>
      <P>
        我们可能不时更新本条款。如有重大变更，我们会在本页面发布更新后的条款并更新&quot;最后更新日期&quot;。变更生效后继续使用本服务即表示您接受修订后的条款。
      </P>

      <H2>十、联系我们</H2>
      <P>如您对本条款有任何疑问，请通过 <MailA>{CONTACT_EMAIL}</MailA> 与我们联系。</P>
    </>
  );
}

export default function TermsOfServicePage() {
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

        <h1 className="text-3xl font-bold mb-2">{t("footer.termsOfService")}</h1>
        <p className="text-sm text-muted-foreground mb-10">
          {locale === "zh" ? `最后更新日期：${LAST_UPDATED}` : `Last updated: ${LAST_UPDATED}`}
        </p>

        {locale === "zh" ? <ChineseContent /> : <EnglishContent />}
      </div>
    </div>
  );
}
