export function FavIcon({ url, title }: { url: string; title?: string }) {
  return (
    <img
      className="h-4 w-4 rounded-full bg-slate-100 shadow-sm"
      width={16}
      height={16}
      src={new URL(url).origin + "/favicon.ico"}
      alt={title}
      onError={(e) => {
        e.currentTarget.src =
          "https://perishablepress.com/wp/wp-content/images/2021/favicon-standard.png";
      }}
    />
  );
}
