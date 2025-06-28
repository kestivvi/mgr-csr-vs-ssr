import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  compress: false,
  compiler: {
    removeConsole: true,
  },
};

export default nextConfig;
