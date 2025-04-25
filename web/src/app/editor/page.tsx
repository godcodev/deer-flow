import ReportEditor from "~/components/editor";

export default function Page() {
  return (
    <main className="flex h-full w-full">
      <div className="flex h-screen flex-auto">
        <ReportEditor />
      </div>
    </main>
  );
}
