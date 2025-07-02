import { loadBibliography } from '../../utils/citations.js';

export async function GET() {
  try {
    const bibliography = await loadBibliography();
    
    // Convert Map to Array for JSON serialization
    const bibliographyArray = Array.from(bibliography.entries());
    
    return new Response(JSON.stringify(bibliographyArray), {
      headers: {
        'Content-Type': 'application/json',
      },
    });
  } catch (error) {
    console.error('Error loading bibliography:', error);
    return new Response(JSON.stringify([]), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }
}