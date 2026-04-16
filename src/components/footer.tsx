export function Footer() {
  return (
    <footer
      style={{
        paddingTop: "2.5rem",
        paddingBottom: "2rem",
        borderTop: "1px solid rgba(0,0,0,0.07)",
        textAlign: "center",
      }}
    >
      <p
        style={{
          fontSize: "0.6875rem",
          letterSpacing: "0.07em",
          color: "rgba(0,0,0,0.25)",
          textTransform: "uppercase",
        }}
      >
        &copy; {new Date().getFullYear()} Isaac Corley
      </p>
    </footer>
  );
}
