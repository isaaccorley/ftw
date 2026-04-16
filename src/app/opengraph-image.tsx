import { generateOgImage, ogSize } from "@/lib/og-image";

export const dynamic = "force-static";
export const alt = "Fields of The World";
export const size = ogSize;
export const contentType = "image/png";

export default function Image() {
  return generateOgImage(
    "Fields of The World",
    "Run field boundary delineation models directly in the browser on satellite imagery.",
  );
}
