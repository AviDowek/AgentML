export async function apiGet(path: string) {
  const base = import.meta.env.VITE_API_URL ?? 'http://localhost:8001'
  const res = await fetch(`${base}${path}`)
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}
