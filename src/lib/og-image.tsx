import { ImageResponse } from "next/og";

export const ogSize = {
  width: 1200,
  height: 630,
};

export function generateOgImage(title: string, subtitle?: string) {
  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "flex-end",
        padding: "60px",
        background: "linear-gradient(135deg, #0a0a0f 0%, #0f1a1a 50%, #0a0a0f 100%)",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: "4px",
          background: "linear-gradient(90deg, #0f766e, #2dd4bf, #0f766e)",
        }}
      />
      <div
        style={{
          position: "absolute",
          top: "60px",
          right: "60px",
          fontSize: "24px",
          color: "#2dd4bf",
          fontWeight: 700,
          letterSpacing: "0.05em",
        }}
      >
        isaac.earth
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "16px",
        }}
      >
        <div
          style={{
            fontSize: title.length > 40 ? "48px" : "56px",
            fontWeight: 700,
            color: "#fafafa",
            lineHeight: 1.2,
            maxWidth: "900px",
          }}
        >
          {title}
        </div>
        {subtitle && (
          <div
            style={{
              fontSize: "24px",
              color: "#94a3b8",
              lineHeight: 1.4,
              maxWidth: "800px",
            }}
          >
            {subtitle}
          </div>
        )}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
            marginTop: "8px",
          }}
        >
          <div
            style={{
              fontSize: "20px",
              color: "#71717a",
            }}
          >
            Isaac Corley
          </div>
        </div>
      </div>
    </div>,
    ogSize,
  );
}
