import React, {useEffect, useState} from 'react'
import {useParams, Link, Routes, Route, NavLink} from 'react-router-dom'
import {apiGet} from '../api/client'
import Data from './Data'
import Experiments from './Experiments'
import Models from './Models'
import DatasetSpecs from './DatasetSpecs'

function TabLink({to, children}:{to:string, children:React.ReactNode}){
  return <NavLink to={to} style={({isActive})=>({marginRight:12, textDecoration:isActive? 'underline':'none'})}>{children}</NavLink>
}

export default function ProjectOverview(){
  const {id} = useParams()
  const [project, setProject] = useState<any>(null)

  useEffect(()=>{
    if(!id) return
    apiGet(`/projects/${id}`).then(setProject).catch(()=>setProject(null))
  },[id])

  if(!project) return <div>Loading project...</div>

  return (
    <div>
      <h1>{project.name}</h1>
      <p>{project.description}</p>

      <nav style={{marginBottom:12}}>
        <TabLink to="data">Data Sources</TabLink>
        <TabLink to="dataset-specs">Dataset Specs</TabLink>
        <TabLink to="experiments">Experiments</TabLink>
        <TabLink to="models">Models</TabLink>
      </nav>

      <Routes>
        <Route path="data" element={<Data/>} />
        <Route path="dataset-specs" element={<DatasetSpecs/>} />
        <Route path="experiments" element={<Experiments/>} />
        <Route path="models" element={<Models/>} />
        <Route path="*" element={<div>Select a tab</div>} />
      </Routes>
    </div>
  )
}
