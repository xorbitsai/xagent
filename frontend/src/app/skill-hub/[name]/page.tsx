import SkillDetailPage from "./page-client";

// Server wrapper for static export: provides a placeholder param so a shell
// HTML is emitted for this dynamic route. FastAPI serves the shell for any real
// name; the client component reads the actual value from the URL via useParams().
export function generateStaticParams() {
  return [{ name: "__shell__" }];
}

export default function Page() {
  return <SkillDetailPage />;
}
