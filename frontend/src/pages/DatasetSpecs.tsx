import React, {useEffect, useState} from 'react'
import { useParams, Link } from 'react-router-dom'
import { apiGet } from '../api/client'

export default function DatasetSpecs(){
  const { id } = useParams()
  const [specs, setSpecs] = useState<any[]>([])

  useEffect(()=>{
    if(!id) return
    apiGet(`/projects/${id}/dataset-specs`).then(setSpecs).catch(()=>setSpecs([]))
  },[id])

  return (
    <div>
      <h2>Dataset Specs</h2>
      <ul>
        {specs.map(s=> (
          <li key={s.id}>{s.name} — {s.description}</li>
        ))}
      </ul>
    </div>
  )
}
