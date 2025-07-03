// Simple bibliography API endpoint
export async function GET() {
  // Return empty bibliography array since citation utilities were removed for simplicity
  return new Response(JSON.stringify([]), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'public, max-age=3600'
    }
  });
}