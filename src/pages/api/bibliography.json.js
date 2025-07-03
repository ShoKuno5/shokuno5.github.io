import { loadBibliography } from '../../utils/citations.js';

export async function GET() {
  try {
    const bibliography = await loadBibliography();
    
    // Convert Map to Array for JSON serialization
    const bibliographyArray = Array.from(bibliography.entries());
    
    return new Response(JSON.stringify(bibliographyArray), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, max-age=3600'
      }
    });
  } catch (error) {
    // Silent error handling - return empty array
    return new Response(JSON.stringify([]), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
}