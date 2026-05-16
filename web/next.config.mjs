/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // We call the FastAPI backend directly from the browser using
  // NEXT_PUBLIC_API_URL — Next.js dev rewrites can buffer SSE in
  // some versions, so direct cross-origin fetch + CORS is safer.
};

export default nextConfig;
