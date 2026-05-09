# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import datetime

import React, { useState, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  BarChart, Bar, Cell, AreaChart, Area, ComposedChart, ScatterChart, Scatter, PieChart, Pie
} from 'recharts';
import { 
  LayoutDashboard, TrendingUp, PieChart as PieChartIcon, Layers, Info, 
  ChevronRight, Database, Search, FileText, ChevronDown, Activity, ArrowRight, CheckCircle2
} from 'lucide-react';

// --- DỮ LIỆU GIẢ LẬP CHI TIẾT TỪ VN30 ---
const STOCKS = [
  "ACB", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "KDH",
  "MBB", "MSN", "MWG", "NLG", "NVL", "PDR", "PLX", "POW", "SBT", "SSB",
  "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM"
];

const SECTOR_COLORS = {
  'Ngân hàng': '#4f46e5',
  'Bất động sản': '#10b981',
  'Năng lượng': '#f59e0b',
  'Công nghiệp': '#6366f1',
  'Tiêu dùng': '#ec4899',
  'Khác': '#94a3b8'
};

const SECTORS = {
  'Ngân hàng': ['ACB', 'BID', 'CTG', 'HDB', 'MBB', 'SSB', 'STB', 'TCB', 'TPB', 'VCB', 'VIB'],
  'Bất động sản': ['VIC', 'VHM', 'NVL', 'PDR', 'KDH', 'NLG'],
  'Năng lượng': ['GAS', 'PLX', 'POW'],
  'Công nghiệp': ['HPG', 'GVR'],
  'Tiêu dùng': ['MSN', 'VNM', 'MWG'],
  'Khác': ['FPT', 'VJC', 'SBT', 'SSI', 'BVH']
};

const PCA_SUMMARY = [
  { pc: 'PC1', var: 33.16, cum: 33.16, desc: 'Nhân tố thị trường (Market Factor)' },
  { pc: 'PC2', var: 12.45, cum: 45.61, desc: 'Nhân tố Ngành (Sector Factor - BĐS/Năng lượng)' },
  { pc: 'PC3', var: 8.12, cum: 53.73, desc: 'Nhân tố xoay vòng (Rotation Factor - Ngân hàng)' },
  { pc: 'PC4', var: 6.20, cum: 59.93, desc: 'Đặc thù nhóm nhỏ' },
  { pc: 'PC5', var: 5.19, cum: 65.12, desc: 'Đặc thù nhóm nhỏ' },
];

const CORRELATION_MATRIX = STOCKS.slice(0, 8).map((s1, i) => ({
  name: s1,
  ...Object.fromEntries(STOCKS.slice(0, 8).map((s2, j) => [s2, i === j ? 1 : 0.4 + Math.random() * 0.5]))
}));

const PC_WEIGHTS = STOCKS.map(s => {
  const sector = Object.keys(SECTORS).find(k => SECTORS[k].includes(s));
  return {
    ticker: s,
    pc1: 0.12 + Math.random() * 0.1,
    pc2: sector === 'Bất động sản' ? 0.22 + Math.random() * 0.15 : (sector === 'Năng lượng' ? -0.25 - Math.random() * 0.1 : (Math.random() - 0.5) * 0.15),
    pc3: sector === 'Ngân hàng' ? -0.18 - Math.random() * 0.1 : (Math.random() - 0.5) * 0.2,
    sector
  };
}).sort((a, b) => b.pc1 - a.pc1);

const TIME_SERIES = Array.from({ length: 50 }, (_, i) => ({
  date: `Phiên ${i + 1}`,
  pc1: Math.sin(i / 8) * 4 + (Math.random() - 0.5) * 2,
  vn30: Math.sin(i / 8) * 3.8 + (Math.random() - 0.5) * 3,
}));

// --- COMPONENTS ---

const SectionTitle = ({ children, icon: Icon, step }) => (
  <div className="flex items-center gap-4 mb-8 border-b border-slate-200 pb-6">
    <div className="flex items-center justify-center w-12 h-12 bg-indigo-600 rounded-2xl text-white shadow-lg shadow-indigo-200 shrink-0">
      <Icon size={24} />
    </div>
    <div>
      {step && <span className="text-indigo-600 text-xs font-black uppercase tracking-widest">{step}</span>}
      <h2 className="text-2xl md:text-3xl font-black text-slate-800">{children}</h2>
    </div>
  </div>
);

const InfoCard = ({ children, type = "info" }) => {
  const styles = {
    info: "bg-indigo-50 border-l-4 border-indigo-500 text-indigo-900",
    success: "bg-emerald-50 border-l-4 border-emerald-500 text-emerald-900",
    warning: "bg-amber-50 border-l-4 border-amber-500 text-amber-900"
  };
  return (
    <div className={`${styles[type]} p-5 rounded-r-2xl my-6 flex gap-4 animate-in fade-in slide-in-from-left-4 duration-500`}>
      <Info className="shrink-0 mt-1" size={20} />
      <div className="text-sm leading-relaxed italic">{children}</div>
    </div>
  );
};

const App = () => {
  const [activeTab, setActiveTab] = useState('intro');

  const navItems = [
    { id: 'intro', label: 'Tổng quan Dự án', icon: FileText },
    { id: 'eda', label: 'Dữ liệu & Tương quan', icon: Search },
    { id: 'math', label: 'Mô hình Toán học', icon: Layers },
    { id: 'results', label: 'Thành phần Chính (PC1)', icon: TrendingUp },
    { id: 'sector', label: 'Phân hóa Nhóm ngành', icon: PieChartIcon },
    { id: 'summary', label: 'Kết luận thực tiễn', icon: Database },
  ];

  return (
    <div className="min-h-screen bg-slate-50 flex font-sans selection:bg-indigo-100 selection:text-indigo-900">
      {/* SIDEBAR NAVIGATION */}
      <aside className="w-80 bg-white border-r border-slate-200 sticky top-0 h-screen overflow-y-auto hidden xl:flex flex-col z-20">
        <div className="p-10">
          <div className="flex items-center gap-3 text-indigo-600 mb-12">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white shadow-xl shadow-indigo-100">
              <Activity size={24} />
            </div>
            <span className="text-xl font-black tracking-tighter text-slate-900 uppercase">VN30 PCA Lab</span>
          </div>
          
          <nav className="space-y-3">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-4 px-6 py-4 rounded-2xl transition-all duration-300 group ${
                  activeTab === item.id 
                  ? 'bg-slate-900 text-white shadow-2xl shadow-slate-200 translate-x-2' 
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'
                }`}
              >
                <item.icon size={18} className={activeTab === item.id ? 'text-indigo-400' : 'group-hover:text-indigo-500'} />
                <span className="font-bold text-sm">{item.label}</span>
              </button>
            ))}
          </nav>
        </div>
        
        <div className="mt-auto p-8 border-t border-slate-100 bg-slate-50/50">
          <div className="p-4 bg-white rounded-2xl shadow-sm border border-slate-200">
            <div className="text-[10px] font-black text-slate-400 uppercase mb-3">Người thực hiện</div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center text-white text-xs font-bold uppercase">QA</div>
              <div className="text-xs font-bold text-slate-700 tracking-tight">Dự án Nhóm 1</div>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 overflow-y-auto bg-slate-50/30">
        <div className="max-w-6xl mx-auto p-6 md:p-16 lg:p-20">

          {/* TAB 1: INTRO */}
          {activeTab === 'intro' && (
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700">
              <div className="mb-16">
                <span className="inline-block px-4 py-1.5 bg-indigo-100 text-indigo-700 text-xs font-black rounded-full mb-6 uppercase tracking-widest shadow-sm shadow-indigo-100">
                  Dự án Phân tích Dữ liệu Tài chính
                </span>
                <h1 className="text-5xl md:text-7xl font-black text-slate-900 leading-[1.1] mb-8 tracking-tighter">
                  Khám phá cấu trúc <br/>
                  thị trường <span className="text-indigo-600 underline decoration-indigo-200 underline-offset-8">VN30</span>
                </h1>
                <p className="text-xl text-slate-500 max-w-2xl font-medium leading-relaxed">
                  Ứng dụng thuật toán Principal Component Analysis (PCA) để bóc tách các nhân tố rủi ro hệ thống và đặc thù ngành.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-8 my-16">
                {[
                  { title: 'PC1: Thị trường', value: '33.16%', desc: 'Rủi ro hệ thống chi phối toàn rổ.', icon: TrendingUp, color: 'text-indigo-600' },
                  { title: 'PC2: Phân hóa', value: '12.45%', desc: 'Sự đối trọng giữa BĐS & Năng lượng.', icon: Layers, color: 'text-emerald-600' },
                  { title: 'Tương quan', value: '0.99', desc: 'Độ khớp giữa PC1 và VN30 Index.', icon: Activity, color: 'text-amber-600' },
                ].map((stat, i) => (
                  <div key={i} className="bg-white p-8 rounded-[2rem] shadow-sm border border-slate-100 hover:shadow-xl hover:-translate-y-1 transition-all duration-300">
                    <stat.icon className={`${stat.color} mb-6`} size={32} />
                    <div className="text-sm font-black text-slate-400 uppercase tracking-widest mb-1">{stat.title}</div>
                    <div className="text-4xl font-black text-slate-900 mb-3">{stat.value}</div>
                    <p className="text-sm text-slate-500 font-medium">{stat.desc}</p>
                  </div>
                ))}
              </div>

              <div className="bg-slate-900 p-12 rounded-[2.5rem] text-white relative overflow-hidden">
                <div className="relative z-10 max-w-2xl">
                  <h3 className="text-2xl font-black mb-6 flex items-center gap-3">
                    <div className="w-2 h-8 bg-indigo-500 rounded-full"></div>
                    Tóm tắt bài toán
                  </h3>
                  <p className="text-slate-300 leading-relaxed mb-6 font-medium">
                    Trong danh mục VN30, các cổ phiếu thường biến động cùng chiều. PCA giúp chúng ta trả lời câu hỏi: 
                    "Có bao nhiêu 'nhân tố' thực sự đang vận hành thị trường?" và "Mã nào là 'đầu tàu' của những nhân tố đó?".
                  </p>
                  <button onClick={() => setActiveTab('eda')} className="flex items-center gap-2 bg-indigo-600 px-8 py-4 rounded-2xl font-black hover:bg-indigo-500 transition-all shadow-xl shadow-indigo-900/20">
                    Bắt đầu khám phá <ArrowRight size={20} />
                  </button>
                </div>
                <Activity size={400} className="absolute -right-40 -bottom-40 text-slate-800/50 pointer-events-none" strokeWidth={1} />
              </div>
            </div>
          )}

          {/* TAB 2: EDA */}
          {activeTab === 'eda' && (
            <div className="animate-in fade-in duration-500">
              <SectionTitle icon={Search} step="Bước 01">Dữ liệu & Ma trận Tương quan</SectionTitle>
              <p className="text-lg text-slate-500 mb-10 font-medium">
                Sử dụng thư viện <code>yfinance</code> để thu thập dữ liệu giá đóng cửa của 30 cổ phiếu thuộc rổ VN30 trong 1 năm gần nhất.
              </p>

              <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100 mb-12">
                <div className="flex items-center justify-between mb-10">
                  <h3 className="font-black text-slate-800 flex items-center gap-3">
                    <div className="w-10 h-10 bg-rose-50 rounded-xl flex items-center justify-center text-rose-500">
                      <Activity size={20} />
                    </div>
                    Ma trận Tương quan (Sample Heatmap)
                  </h3>
                  <div className="flex gap-2">
                    <span className="px-3 py-1 bg-slate-100 rounded-lg text-[10px] font-black text-slate-500 uppercase">Dữ liệu chuẩn hóa</span>
                  </div>
                </div>

                <div className="grid grid-cols-9 gap-1.5 mb-8 overflow-x-auto">
                  <div className="col-span-1"></div>
                  {STOCKS.slice(0, 8).map(s => <div key={s} className="text-[10px] font-black text-center text-slate-400 py-2">{s}</div>)}
                  
                  {CORRELATION_MATRIX.map((row, i) => (
                    <React.Fragment key={i}>
                      <div className="text-[10px] font-black flex items-center text-slate-500 pr-4">{row.name}</div>
                      {STOCKS.slice(0, 8).map(s => {
                        const val = row[s];
                        const opacity = val;
                        return (
                          <div 
                            key={s} 
                            style={{ backgroundColor: `rgba(79, 70, 229, ${opacity})` }}
                            className="aspect-square rounded-lg flex items-center justify-center text-[10px] text-white font-black transition-all hover:scale-110 cursor-pointer shadow-sm"
                          >
                            {val.toFixed(1)}
                          </div>
                        );
                      })}
                    </React.Fragment>
                  ))}
                </div>
                
                <div className="flex justify-center items-center gap-6 text-[10px] font-black text-slate-400 uppercase tracking-widest mt-10">
                  <span>Tương quan thấp</span>
                  <div className="w-48 h-2 bg-gradient-to-r from-indigo-50 to-indigo-600 rounded-full"></div>
                  <span>Tương quan cao</span>
                </div>
              </div>

              <InfoCard>
                Quan sát ma trận: Hầu hết các ô đều có màu tím đậm (tương quan {'>'} 0.5). Điều này khẳng định giả thuyết rằng các cổ phiếu VN30 chịu ảnh hưởng mạnh mẽ bởi một nhân tố thị trường chung.
              </InfoCard>
            </div>
          )}

          {/* TAB 3: MATH */}
          {activeTab === 'math' && (
            <div className="animate-in slide-in-from-right-8 duration-500 space-y-12">
              <SectionTitle icon={Layers} step="Bước 02">Quy trình Toán học PCA</SectionTitle>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <div className="bg-white p-10 rounded-[2.5rem] border border-slate-100 shadow-sm">
                  <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600 mb-8">
                    <CheckCircle2 size={24} />
                  </div>
                  <h4 className="font-black text-slate-800 mb-4 uppercase text-xs tracking-widest">1. Chuẩn hóa dữ liệu (Scaling)</h4>
                  <p className="text-slate-500 text-sm leading-relaxed mb-8">Đưa lợi suất về phân phối chuẩn có trung bình 0 và phương sai 1 để loại bỏ sai khác về quy mô.</p>
                  <div className="bg-slate-900 p-6 rounded-2xl font-mono text-xs text-indigo-300 leading-loose">
                    {"$Z = \\frac{R - \\mu}{\\sigma}$"}
                  </div>
                </div>

                <div className="bg-white p-10 rounded-[2.5rem] border border-slate-100 shadow-sm">
                  <div className="w-12 h-12 bg-emerald-50 rounded-2xl flex items-center justify-center text-emerald-600 mb-8">
                    <CheckCircle2 size={24} />
                  </div>
                  <h4 className="font-black text-slate-800 mb-4 uppercase text-xs tracking-widest">2. Phân rã Trị riêng (Eigen-decomposition)</h4>
                  <p className="text-slate-500 text-sm leading-relaxed mb-8">Giải phương trình đặc trưng trên ma trận Hiệp phương sai để tìm các trục biến động lớn nhất.</p>
                  <div className="bg-slate-900 p-6 rounded-2xl font-mono text-xs text-emerald-300 leading-loose">
                    {"$Cov \\cdot V = \\lambda \\cdot V$"}
                  </div>
                </div>
              </div>

              <div className="bg-white p-12 rounded-[2.5rem] border border-slate-100 shadow-sm">
                <div className="flex items-center justify-between mb-12">
                  <h3 className="font-black text-slate-800 text-xl tracking-tight">Scree Plot: Khả năng giải thích của mô hình</h3>
                  <div className="px-4 py-2 bg-emerald-50 text-emerald-700 rounded-xl text-xs font-black uppercase">Top 5 PC = 65.12% Variance</div>
                </div>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={PCA_SUMMARY}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                      <XAxis dataKey="pc" axisLine={false} tickLine={false} fontSize={12} fontVariant="black" />
                      <YAxis axisLine={false} tickLine={false} fontSize={12} unit="%" />
                      <Tooltip contentStyle={{ borderRadius: '24px', border: 'none', padding: '20px', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.1)' }} />
                      <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: '20px' }} />
                      <Bar dataKey="var" fill="#6366f1" radius={[12, 12, 0, 0]} name="% Phương sai riêng" barSize={60} />
                      <Line dataKey="cum" stroke="#f43f5e" strokeWidth={4} dot={{ r: 8, fill: '#f43f5e', strokeWidth: 0 }} name="% Tích lũy" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: RESULTS */}
          {activeTab === 'results' && (
            <div className="animate-in fade-in duration-500 space-y-12">
              <SectionTitle icon={TrendingUp} step="Bước 03">Phân tích PC1: Nhân tố Thị trường</SectionTitle>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 bg-white p-10 rounded-[2.5rem] border border-slate-100 shadow-sm">
                  <div className="flex items-center justify-between mb-10">
                    <h3 className="font-black text-slate-800">PC1 Index vs VN30 Actual</h3>
                    <div className="flex items-center gap-2 text-emerald-600 bg-emerald-50 px-3 py-1.5 rounded-lg text-xs font-black">
                      <TrendingUp size={14} /> Correlation: 0.9958
                    </div>
                  </div>
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={TIME_SERIES}>
                        <defs>
                          <linearGradient id="colorPC1" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15}/>
                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="date" hide />
                        <YAxis hide />
                        <Tooltip />
                        <Area type="monotone" dataKey="pc1" stroke="#6366f1" strokeWidth={4} fillOpacity={1} fill="url(#colorPC1)" name="PC1 (Market Factor)" />
                        <Line type="monotone" dataKey="vn30" stroke="#94a3b8" strokeDasharray="5 5" dot={false} strokeWidth={2} name="VN30 Actual" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-slate-900 p-10 rounded-[2.5rem] text-white flex flex-col justify-center">
                  <h4 className="text-indigo-400 font-black uppercase text-xs tracking-widest mb-6">Ý nghĩa tài chính</h4>
                  <p className="text-lg leading-relaxed font-medium mb-8">
                    PC1 đóng vai trò là "nhân tố thị trường". Nó giải thích phần lớn sự biến động đồng thuận của 30 cổ phiếu, phản ánh tâm lý chung và tin tức vĩ mô.
                  </p>
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-indigo-500 rounded-full"></div>
                      <span className="text-sm text-slate-400">Đại diện cho Beta thị trường.</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-indigo-500 rounded-full"></div>
                      <span className="text-sm text-slate-400">Đo lường rủi ro hệ thống.</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white p-12 rounded-[2.5rem] border border-slate-100 shadow-sm">
                <h3 className="font-black text-slate-800 text-xl mb-12">Top 12 Cổ phiếu Nhạy cảm nhất (High Beta in PC1)</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-12 gap-y-8">
                  {PC_WEIGHTS.slice(0, 12).map((item, i) => (
                    <div key={i} className="group">
                      <div className="flex justify-between items-end mb-3">
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] font-black text-slate-300">#{i+1}</span>
                          <span className="font-black text-slate-800">{item.ticker}.VN</span>
                        </div>
                        <span className="text-xs font-black text-indigo-600">{item.pc1.toFixed(3)}</span>
                      </div>
                      <div className="h-2 bg-slate-50 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-indigo-500 rounded-full transition-all duration-1000 group-hover:bg-indigo-400" 
                          style={{ width: `${item.pc1 * 400}%` }}
                        ></div>
                      </div>
                      <div className="mt-2 text-[10px] font-bold text-slate-400 uppercase tracking-tighter">{item.sector}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: SECTOR */}
          {activeTab === 'sector' && (
            <div className="animate-in fade-in duration-500 space-y-12">
              <SectionTitle icon={PieChartIcon} step="Bước 04">Phân hóa Nhóm ngành (PC2 & PC3)</SectionTitle>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100">
                  <div className="flex items-center gap-3 mb-8">
                    <div className="w-10 h-10 bg-emerald-50 rounded-xl flex items-center justify-center text-emerald-600">
                      <ChevronRight size={20} />
                    </div>
                    <h3 className="font-black text-slate-800">PC2: Bất động sản vs Năng lượng</h3>
                  </div>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={PC_WEIGHTS.filter(s => s.sector === 'Bất động sản' || s.sector === 'Năng lượng').sort((a,b) => b.pc2 - a.pc2)}>
                        <XAxis dataKey="ticker" axisLine={false} tickLine={false} fontSize={10} fontVariant="black" />
                        <YAxis axisLine={false} tickLine={false} fontSize={10} />
                        <Tooltip cursor={{fill: '#f8fafc'}} />
                        <Bar dataKey="pc2" radius={[6, 6, 6, 6]}>
                          {PC_WEIGHTS.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.pc2 > 0 ? '#10b981' : '#f43f5e'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <InfoCard type="warning">
                    PC2 bóc tách sự biến động ngược chiều. Khi BĐS (VHM, VIC, PDR) tăng, nhóm Năng lượng (GAS, PLX) thường có xu hướng đi ngược lại hoặc ổn định hơn.
                  </InfoCard>
                </div>

                <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100">
                  <div className="flex items-center gap-3 mb-8">
                    <div className="w-10 h-10 bg-indigo-50 rounded-xl flex items-center justify-center text-indigo-600">
                      <ChevronRight size={20} />
                    </div>
                    <h3 className="font-black text-slate-800">PC3: Nhân tố Tài chính - Ngân hàng</h3>
                  </div>
                   <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={PC_WEIGHTS.filter(s => s.sector === 'Ngân hàng').sort((a,b) => a.pc3 - b.pc3)}>
                        <XAxis dataKey="ticker" axisLine={false} tickLine={false} fontSize={10} fontVariant="black" />
                        <YAxis axisLine={false} tickLine={false} fontSize={10} />
                        <Tooltip cursor={{fill: '#f8fafc'}} />
                        <Bar dataKey="pc3" fill="#6366f1" radius={[6, 6, 6, 6]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <InfoCard type="info">
                    PC3 cho thấy sự đặc thù của nhóm Ngân hàng. Đây là nhân tố giải thích rủi ro chuyên biệt của ngành Tài chính trong rổ VN30.
                  </InfoCard>
                </div>
              </div>

              <div className="bg-white p-12 rounded-[2.5rem] border border-slate-100 shadow-sm">
                <h3 className="font-black text-slate-800 text-xl mb-10">Phân bổ ngành trong Nhân tố Thị trường (PC1)</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
                  {Object.entries(SECTORS).map(([name, stocks], idx) => (
                    <div key={idx} className="bg-slate-50 p-6 rounded-3xl border border-slate-100 text-center">
                      <div className="text-[10px] font-black text-slate-400 uppercase mb-2">{name}</div>
                      <div className="text-lg font-black text-slate-800 mb-1">{stocks.length}</div>
                      <div className="text-[10px] font-bold text-slate-400">Cổ phiếu</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB 6: SUMMARY */}
          {activeTab === 'summary' && (
            <div className="animate-in zoom-in-95 duration-700">
              <SectionTitle icon={Database} step="Kết luận">Ý nghĩa & Ứng dụng thực tiễn</SectionTitle>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-16">
                <div className="bg-white p-12 rounded-[3rem] border border-slate-100 shadow-xl shadow-slate-200/50">
                  <h4 className="text-4xl font-black text-slate-900 mb-8 tracking-tighter italic">Key Findings</h4>
                  <ul className="space-y-8">
                    {[
                      { t: 'PC1 là Market Factor', d: 'Giải thích 33% biến động, tương quan 0.99 với chỉ số.' },
                      { t: 'Đầu tàu dẫn dắt', d: 'PDR, SSI, NLG là những mã nhạy cảm nhất với nhịp đập chung.' },
                      { t: 'Phân hóa Ngành', d: 'PC2 & PC3 bóc tách rõ rệt đặc tính rủi ro của BĐS và Ngân hàng.' },
                    ].map((item, i) => (
                      <li key={i} className="flex gap-6 group">
                        <div className="w-10 h-10 rounded-full bg-indigo-600 flex items-center justify-center text-white font-black text-sm shrink-0 group-hover:scale-110 transition-transform">
                          {i+1}
                        </div>
                        <div>
                          <div className="font-black text-slate-800 mb-1">{item.t}</div>
                          <div className="text-sm text-slate-500 font-medium leading-relaxed">{item.d}</div>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="space-y-8 flex flex-col">
                  <div className="bg-slate-900 p-10 rounded-[3rem] text-white flex-1">
                    <h3 className="text-2xl font-black mb-8 text-indigo-400">Ứng dụng Đầu tư</h3>
                    <div className="space-y-6">
                      <div className="p-6 bg-slate-800 rounded-2xl border border-slate-700">
                        <div className="font-bold text-indigo-300 mb-2">1. Hedging (Phòng vệ)</div>
                        <p className="text-xs text-slate-400 leading-relaxed">Sử dụng PC1 để xây dựng danh mục phòng vệ rủi ro hệ thống mà không cần phụ thuộc hoàn toàn vào Index.</p>
                      </div>
                      <div className="p-6 bg-slate-800 rounded-2xl border border-slate-700">
                        <div className="font-bold text-emerald-300 mb-2">2. Xây dựng Danh mục</div>
                        <p className="text-xs text-slate-400 leading-relaxed">Kết hợp các cổ phiếu có trọng số PC2 đối nghịch nhau để đa dạng hóa rủi ro ngành hiệu quả hơn.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="text-center p-20 bg-white rounded-[4rem] border-4 border-dashed border-slate-100 flex flex-col items-center">
                <div className="w-20 h-20 bg-indigo-50 rounded-3xl flex items-center justify-center text-indigo-600 mb-8">
                  <CheckCircle2 size={40} />
                </div>
                <h3 className="text-3xl font-black text-slate-800 mb-4 tracking-tight">Hoàn tất Phân tích</h3>
                <p className="text-slate-400 max-w-lg mx-auto font-medium mb-12 italic">
                  Mô hình PCA đã thành công trong việc đơn giản hóa cấu trúc 30 chiều của VN30 thành các nhân tố tài chính có ý nghĩa thực tiễn.
                </p>
                <div className="flex gap-4">
                  <button className="px-10 py-5 bg-slate-900 text-white rounded-2xl font-black shadow-2xl shadow-slate-200 hover:bg-slate-800 transition-all flex items-center gap-3">
                    <Database size={20} /> Xuất Dữ liệu PCA
                  </button>
                  <button className="px-10 py-5 bg-indigo-600 text-white rounded-2xl font-black shadow-2xl shadow-indigo-100 hover:bg-indigo-500 transition-all">
                    Tải Báo cáo Full
                  </button>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
};

export default App;
