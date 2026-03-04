import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const publicDir = path.join(process.cwd(), "public");
    const entries = fs.readdirSync(publicDir, { withFileTypes: true });
    const exts = [".jpg", ".jpeg", ".png", ".webp", ".gif"];

    const images = entries
      .filter((entry) => entry.isFile() && exts.includes(path.extname(entry.name).toLowerCase()))
      .map((entry) => `/${entry.name}`);

    return NextResponse.json({ images });
  } catch (error) {
    return NextResponse.json({ images: [] });
  }
}

